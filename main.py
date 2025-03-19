import streamlit as st
import pandas as pd
import os
import datetime
from io import BytesIO
import plotly.express as px
from opt import execute_optimization, summarize_schedule, output_table, add_minutes_to_datetime


### DataFrameからxlsxファイルに変換
def df_to_xlsx(df_data, df_time):
    byte_xlsx = BytesIO()
    # with pd.ExcelWriter(byte_xlsx, engine="xlsxwriter") as writer:
    with pd.ExcelWriter(byte_xlsx, engine="openpyxl") as writer:
        if type(df_data) is list:
            for i, df_cur in enumerate(df_data, start=1):
                df_cur.to_excel(writer, sheet_name='data' + str(i))
        else:
            df_data.to_excel(writer, sheet_name='data')
        df_time.to_excel(writer, sheet_name='setting', index=False)
    return byte_xlsx.getvalue()

### ページ1：データ読込
def read_data():
    st.markdown(
        """
        #### 1: スケジュールで使用するExcelファイル(xlsx)の読込
        """
    )
    input_mode = st.radio(
        "読込方法を選択：",
        ('サンプルデータを読み込む', 'データをアップロードする'), horizontal=True
    )
    if input_mode == 'データをアップロードする':
        input_file = st.file_uploader("Excelファイルをドラッグ＆ドロップ，または[Browse files]から選択してください", type=['xlsx'])
        if input_file is not None:
            try:
                df_data = pd.read_excel(input_file, sheet_name='data', index_col='ID')
                df_time = pd.read_excel(input_file, sheet_name='setting', header=None, names=['時刻'])
            except:
                st.write('アップロードしたファイルに誤りがあります．ファイルを確認してください．')
                return -1
            # 読み込みデータをSessionStateに保存
            df_data['納期'] = pd.to_datetime(df_data['納期']).dt.date
            st.session_state.df_data = df_data
            st.session_state.df_time = df_time
            # st.session_state.tt = df_time
            st.session_state.is_loaded = True
            st.session_state.is_solved = False
    else:
        # カレントフォルダ内のsampledataフォルダからExcelファイル一覧を取得
        sampledata_folder = "sampledata"
        file_list = sorted([f for f in os.listdir(sampledata_folder) if f.lower().endswith(".xlsx")])

        with st.container():
            col = st.columns([0.35, 0.15, 0.5], vertical_alignment='bottom')
            with col[0]:
                # ファイル選択用の選択ボックスを表示
                input_file = st.selectbox("Excelファイルを選択してください", file_list)
            with col[1]:
                if st.button('読込'):
                    # 選択されたExcelファイルをDataFrameとして読み込み
                    file_path = os.path.join(sampledata_folder, input_file)
                    try:
                        df_data = pd.read_excel(file_path, sheet_name='data', index_col='ID')
                        df_time = pd.read_excel(file_path, sheet_name='setting', header=None, names=['時刻'])
                    except:
                        st.write('サンプルファイルに誤りがあります．別のファイルを選択してください．')
                        return -1
                    # 読み込みデータをSessionStateに保存
                    df_data['納期'] = pd.to_datetime(df_data['納期']).dt.date
                    st.session_state.df_data = df_data
                    st.session_state.df_time = df_time
                    # st.session_state.tt = df_time
                    st.session_state.is_loaded = True
                    st.session_state.is_solved = False
            with col[2]:
                # 選択されたExcelファイルをDataFrameとして読み込み
                file_path = os.path.join(sampledata_folder, input_file)
                try:
                    df_data = pd.read_excel(file_path, sheet_name='data', index_col='ID')
                    df_time = pd.read_excel(file_path, sheet_name='setting', header=None, names=['時刻'])
                except:
                    st.write('サンプルファイルに誤りがあります．別のファイルを選択してください．')
                    return -1
                file_name = input_file
                st.download_button(label="📥 サンプルファイルのダウンロード",
                               data=df_to_xlsx(df_data, df_time), file_name=file_name)

    if st.session_state.is_loaded:
        st.write("読み込んだデータ:")
        st.dataframe(st.session_state.df_data)   # 読み込んだDataFrameを表示
        st.dataframe(st.session_state.df_time)

### ページ2：スケジュール作成
def make_schedule():
    def output_schedule(dfs_schedule):  # dfs：作業員・モデルごとのdf
        # #ガントチャートの描画関数
        def draw_schedule(df):
            color_scale = {
                "Before": "rgb(255,153,178)",  # 前段取の色を赤に設定
                "After": "rgb(153,229,255)"  # 加工の色を青に設定
            }

            fig = px.timeline(df, x_start="開始時刻", x_end="終了時刻", y="順番", color="前後", color_discrete_map=color_scale)
            fig.update_traces(marker=dict(line=dict(width=1, color='black')), selector=dict(type='bar'))  # 棒の輪郭を黒線で付ける
            fig.update_yaxes(autorange="reversed")  # 縦軸を降順に変更
            # fig.update_traces(textposition='inside', insidetextanchor='middle') # px.timelineの引数textを置く位置を内側の中央に変更

            # ラベルを手動で配置するためのannotationsを作成
            annotations = []
            for row in df.itertuples():
                # 仕事IDを棒の左側に配置
                if row.前後 == "Before":
                    id_text = f'<b>{row.ID}</b>' if pd.notna(row.優先) else str(row.ID)
                    annotation_work_id = dict(
                        x=row.開始時刻 + datetime.timedelta(minutes=-7), y=row.順番,
                        text=id_text, showarrow=False
                    )
                    annotations.append(annotation_work_id)

                # 作業時間を棒の中央に配置
                annotation_work_time = dict(
                    x=row.開始時刻 + (row.終了時刻 - row.開始時刻) / 2, y=row.順番,   # datetime同志の加算は不可．datetime + timedeltaは〇
                    text=str((row.終了時刻 - row.開始時刻).seconds // 60), showarrow=False,
                    font=dict(size=10)
                )
                annotations.append(annotation_work_time)
            fig.update_layout(annotations=annotations)  # annotationsを設定

            # # 昼休みなどの時間に縦線を付ける
            max_y = len(set(df['ID'].to_list())) + 0.5      # IDの個数
            today = pd.Timestamp.today().normalize()        # 本日の時刻0:00
            for row in st.session_state.df_time.itertuples():
                x = pd.Timestamp.combine(today, row.時刻)
                fig.add_shape(
                    dict(
                        type="line",
                        x0=x, x1=x, y0=0.5, y1=max_y,
                        line=dict(color="red", width=1)
                    )
                )
            fig.update_layout(title_font_size=20)
            # fig.show()        # Google Colabではこちら
            st.plotly_chart(fig, theme=None)  # theme=None: デザインをstreamlit版にしない

        # ガントチャート描画
        key_dic = dict()        # キー：モデル，値：作業員リスト
        for (i, mode) in dfs_schedule:
            if mode in key_dic:
                key_dic[mode].append(i)
            else:
                key_dic[mode] = [i]
        for mode in sorted(key_dic.keys()):     # モデル順をソート
            st.markdown(f'##### モデル{mode}')
            for i in sorted(key_dic[mode]):     # 作業員をソート
                st.write(f'作業員{i}のスケジュール')
                draw_schedule(dfs_schedule[i, mode])        # ガントチャートの表示

    st.markdown(
        """
        #### 2: 最適スケジュールの作成
        """
    )
    if 'df_data' not in st.session_state:
        st.write("まず，データを読み込んで下さい．")
        return

    df_data = st.session_state.df_data
    df_time = st.session_state.df_time
    with st.expander(f"読込データ:（折り畳みを解除して確認できます）"):
        st.dataframe(df_data)
        st.dataframe(df_time)

    # スケジュール作成実行
    if st.button("スケジュール作成実行"):
        # 各モデル，各作業員のスケジュールを立案する
        dfs_schedule = execute_optimization(df_data, df_time)   # 各作業員・モデルのスケジュール(df)
        if dfs_schedule is None:
            st.write('最適なスケジュールが見つかりませんでした．入力ファイルを確認してください．')
        else:
            st.session_state.is_solved = True
            st.session_state.dfs_schedule = dfs_schedule    # ガントチャート表示用df
            st.session_state.df_summarized = summarize_schedule(dfs_schedule, df_data) # 割当結果表df
            st.session_state.df_table = output_table(st.session_state.df_summarized, df_data) # モデルの比較表

    if st.session_state.is_solved:
        # ガントチャートの出力
        output_schedule(st.session_state.dfs_schedule)
        st.write('割当結果')
        st.dataframe(st.session_state.df_summarized)
        st.write('モデルの比較')
        st.dataframe(st.session_state.df_table)

        # 結果のダウンロードボタン
        file_name = 'result-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.xlsx'
        st.download_button(label="📥 結果のダウンロード", data=df_to_xlsx(df_data=st.session_state.df_summarized, df_time=st.session_state.df_time), file_name=file_name)

### ページ3：設定変更
def change_settings():
    st.markdown(
        """
        #### *: スケジュールで使用する設定の変更
        """
    )
    if 'df_time' not in st.session_state:
        st.write("まず，データを読み込んで下さい．")
        return

    with st.container():
        st.dataframe(st.session_state.df_time)
        col = st.columns(3, vertical_alignment='center')
        with col[0]:
            st.write('各時刻の設定')
        with col[2]:
            file_name = 'setting-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.xlsx'
            st.download_button(label="📥 設定値のダウンロード", data=df_to_xlsx(st.session_state.df_data, st.session_state.df_time), file_name=file_name)

        col = st.columns(3)
        labels = ['AM開始時刻', 'AM終了時刻', 'PM1開始時刻', 'PM1終了時刻', 'PM2開始時刻', 'PM2終了時刻']
        for i in range(3):
            with col[i]:
                for j in range(2):
                    st.session_state.df_time.loc[2 * i + j, '時刻'] = st.time_input(labels[2 * i + j], st.session_state.df_time['時刻'][2 * i + j], step=60)
    st.dataframe(st.session_state.df_time)

### メインプログラム
def main():
    # 画面全体の設定
    st.set_page_config(
        page_title="スケジュール自動作成アプリ",
        page_icon="🖥️",
        layout="centered",
    )

    # SessionStateの設定
    if 'is_loaded' not in st.session_state: st.session_state.is_loaded = False
    if 'is_solved' not in st.session_state: st.session_state.is_solved = False

    tab1, tab2, tab3 = st.tabs(["1: データ読込", "2: スケジュール作成", "*: 設定変更"])
    with tab1:  read_data()
    with tab2:  make_schedule()
    with tab3:  change_settings()

if __name__ == "__main__":
    main()
