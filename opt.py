from mip import *
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import pandas as pd

# 時間・時刻．以下の内容はサンプル．実際には，関数 preparation で設定
TT = [0, 210, 260, 420, 430, 530]  # 各時刻．AM開始，AM終了，PM1開始，PM1終了，PM2開始，PM2終了
T = dict(zip([1, 12, 2, 23, 3], [TT[i + 1] - TT[i] for i in range(5)]))  # 各時間．AM，昼休み，PM1，PM休み，PM2


### 前準備
def preparation(start=None):
    global TT, T
    tt = st.session_state.tt['時刻']
    today = datetime.now().date()
    # TT[0]=0とした，分単位時刻
    TT = [(datetime.combine(today, tt[i]) - datetime.combine(today, tt[0])).seconds // 60 for i in range(6)]
    T = dict(zip([1, 12, 2, 23, 3], [TT[i + 1] - TT[i] for i in range(5)]))


##最適化
def solve_model1(df):
    def val(variable):
        return int(variable.x + 0.1)

    # #定数用のデータの作成
    J = df.index
    a, bb, n = df['自動前段取'], df['自動加工'], df['セット数']
    b = {j: bb[j] / n[j] for j in J}
    prioAM, prioToday = df['午前優先'], df['当日優先']
    alpha = [1000, 100, 0.001]  # 重み．午前優先，当日優先，早く終わる価値
    # print(f'{J = }')
    # print(f'{a = }')
    # print(f'{b = }')
    # print(f'{n = }')
    # print(f'{prioAM = }')
    # print(f'{prioToday = }')

    #空問題の作成
    model = Model('Schedule')

    # #決定変数の作成
    x, v, y, w, z = {}, {}, {}, {}, {}
    for j in J:
        x[j] = model.add_var(f'x{j}', var_type='B')  # AM中に開始する仕事
        v[j] = model.add_var(f'v{j}', var_type='B')  # AM中に開始して，昼休みも作業する仕事
        y[j] = model.add_var(f'y{j}', var_type='B')  # PM1中に仕事開始する仕事
        w[j] = model.add_var(f'w{j}', var_type='B')  # PM1までに開始して，午後休みも作業する仕事
        z[j] = model.add_var(f'z{j}', var_type='B')  # PM2中に開始する仕事（完了もする）
    t1 = model.add_var('t1')  # v=1の仕事を昼休みに行う時間
    t2 = model.add_var('t2')  # w=1の仕事を午後休みに行う時間
    xi = model.add_var('xi', var_type='B')  # v=1の仕事がPM1で終わらないケースになるか

    # #制約条件の追加

    # #昼休み終了まで
    model += xsum(a[j] * x[j] + n[j] * b[j] * (x[j] - v[j]) for j in J) <= T[1]
    model += xsum(v[j] for j in J) <= 1  # v[j] = 1となる変数は，0個か1個
    for j in J:
        model += v[j] <= x[j]
        model += t1 <= T[12] * (1 - v[j]) + b[j] * v[j]
    model += t1 <= T[12]
    model += T[1] + t1 <= xsum((a[j] + n[j] * b[j]) * x[j] for j in J)

    # 午前最後の仕事が，午後1でも終わらない場合の処理．ξ=1となる．
    model += xsum((a[j] + b[j] * n[j]) * x[j] for j in J) <= T[1] + t1 + T[2] + T[3] * xi
    for j in J:
        model += y[j] <= 1 - xi  # ξ=1のときは，午後1に仕事を始めない
        model += w[j] <= 1 - xi + v[j]  # ただし，v=1の仕事を午後休みに行うのはOK

    # #午後1終了まで
    model += xsum(a[j] * y[j] + n[j] * b[j] * (y[j] - w[j]) for j in J) <= T[2]
    model += xsum((a[j] + n[j] * b[j]) * x[j] + a[j] * y[j] + n[j] * b[j] * (y[j] - w[j]) for j in J) <= T[1] + t1 + T[
        2]
    model += xsum(w[j] for j in J) <= 1  # w[j] = 1となる変数は，0個か1個
    for j in J:
        model += w[j] <= y[j] + v[j]  # 午後休みに仕事できるのは，午後1に始めた仕事かv=1の仕事
        model += t2 <= T[23] * (1 - w[j]) + b[j] * w[j]
    model += t2 <= T[23]
    model += T[1] + t1 + T[2] + t2 <= xsum((a[j] + n[j] * b[j]) * (x[j] + y[j]) for j in J)

    # #終了まで
    model += xsum((a[j] + n[j] * b[j]) * z[j] for j in J) <= T[3]
    model += xsum((a[j] + n[j] * b[j]) * (x[j] + y[j] + z[j]) for j in J) <= T[1] + t1 + T[2] + t2 + T[3]

    # ##その他
    for j in J:
        model += x[j] + y[j] + z[j] <= 1

    # #目的関数の設定
    model.objective = maximize(xsum(
        alpha[0] * prioAM[j] * x[j] + (alpha[1] * (prioAM[j] + prioToday[j]) + 1) * (x[j] + y[j] + z[j]) for j in J))
    # model.write("model.lp")

    # #最適化の実行
    model.verbose = 0  # 実行過程の非表示
    status = model.optimize()

    # #最適化の結果出力
    if status == OptimizationStatus.OPTIMAL:
        df['x'] = [val(x[j]) for j in J]
        df['v'] = [val(v[j]) for j in J]
        df['y'] = [val(y[j]) for j in J]
        df['w'] = [val(w[j]) for j in J]
        df['z'] = [val(z[j]) for j in J]
        # print(df)
        # print(f'{val(t1) = }, {val(t2) = }, {val(xi) = }')
        return df
    else:
        # print('最適解が求まりませんでした。')
        return None


# 時間経過後の時刻を返す関数
def add_minutes_to_datetime(minute_to_add):
    # 指定された日時をdatetimeオブジェクトに変換
    today = datetime.now().date()
    dt = datetime.combine(today, st.session_state.tt['時刻'][0])
    # dt = pd.to_datetime(f"{now.year}-{now.month}-{now.day}-{START_TIME}", format='%Y-%m-%d-%H:%M')
    # 指定された分の時間を加算
    return dt + timedelta(minutes=float(minute_to_add))


### 最適化の結果から，生産時間を算出
def construct_schedule(df):
    # 1つの仕事の開始・終了時刻を計算し，resultに書き込み，cur_timeを更新
    # j: ID，t：作業時間，ty：タイプ（Before/After）, order：通し番号
    def write_job(t, ty, order):
        nonlocal result, cur_time

        result["仕事名"].append(f"Task{row.Index}_{ty}")
        result["ID"].append(row.Index)
        result["開始時刻"].append(add_minutes_to_datetime(cur_time))
        cur_time += t
        result["終了時刻"].append(add_minutes_to_datetime(cur_time))
        result["順番"].append(order)
        result["前後"].append(ty)
        if row.午前優先:
            result["優先"].append('午前')
        elif row.当日優先:
            result["優先"].append('当日')
        else:
            result["優先"].append('　　')

    #仕事一覧とその仕事の開始時刻、終了時刻
    result = {"仕事名": [], "ID": [], "開始時刻": [], "終了時刻": [], "順番": [], "前後": [], "優先": []}
    d = df.sort_values(by=['午前優先', '当日優先'], ascending=False)
    order = 1
    cur_time = 0

    # AM1で終わる仕事
    for row in d.itertuples():
        if row.x > 0 and row.v == 0:
            write_job(row.自動前段取, 'Before', order)
            for _ in range(row.セット数):
                write_job(row.自動加工 / row.セット数, 'After', order)
            order += 1
    # AM1で開始するが終わらない仕事
    for row in d.itertuples():
        if row.v > 0:
            write_job(row.自動前段取, 'Before', order)
            b = row.自動加工 / row.セット数
            n = min(int((TT[1] - cur_time) / b) + 1, row.セット数)
            for _ in range(n):
                write_job(b, 'After', order)

            cur_time = max(cur_time, TT[2])  # v=1の仕事の昼休み分が終わる時刻
            if n < row.セット数:
                if cur_time + (row.セット数 - n) * b <= TT[3]:  # v=1の仕事がPM1内で終わる場合
                    for _ in range(row.セット数 - n):
                        write_job(b, 'After', order)
                else:
                    n2 = min(int((TT[3] - cur_time) / b) + 1, row.セット数 - n)
                    for _ in range(n2):
                        write_job(b, 'After', order)
                    cur_time = max(cur_time, TT[4])
                    if n + n2 < row.セット数:
                        for _ in range(row.セット数 - n - n2):
                            write_job(b, 'After', order)
            order += 1
    # PM1内で開始して終わる仕事
    cur_time = max(cur_time, TT[2])  # v=1の仕事が終わる時刻
    for row in d.itertuples():
        if row.y > 0 and row.w == 0:
            write_job(row.自動前段取, 'Before', order)
            for _ in range(row.セット数):
                write_job(row.自動加工 / row.セット数, 'After', order)
            order += 1
    # PM1で開始するが，PM1で終わらない仕事
    for row in d.itertuples():
        if row.y > 0 and row.w > 0:
            write_job(row.自動前段取, 'Before', order)
            b = row.自動加工 / row.セット数
            n = min(int((TT[3] - cur_time) / b) + 1, row.セット数)
            for _ in range(n):
                write_job(b, 'After', order)

            cur_time = max(cur_time, TT[4])  # v=1の仕事の昼休み分が終わる時刻
            if n < row.セット数:
                for _ in range(row.セット数 - n):
                    write_job(b, 'After', order)
            order += 1
    # PM2で開始して終わる仕事
    cur_time = max(cur_time, TT[4])  # w=1(or v=1)の仕事が終わる時刻
    for row in d.itertuples():
        if row.z > 0:
            write_job(row.自動前段取, 'Before', order)
            for _ in range(row.セット数):
                write_job(row.自動加工 / row.セット数, 'After', order)
            order += 1
    return pd.DataFrame(result)


def output_schedule(df_opt, df_schedule):
    ##ガントチャートの描画関数
    def draw_schedule(data):
        color_scale = {
            "Before": "rgb(255,153,178)",  # 前段取の色を赤に設定
            "After": "rgb(153,229,255)"  # 加工の色を青に設定
        }

        fig = px.timeline(data, x_start="開始時刻", x_end="終了時刻", y="順番", color="前後",
                          color_discrete_map=color_scale)
        fig.update_traces(marker=dict(line=dict(width=1, color='black')), selector=dict(type='bar'))  # 棒の輪郭を黒線で付ける
        fig.update_yaxes(autorange="reversed")  #縦軸を降順に変更
        # fig.update_traces(textposition='inside', insidetextanchor='middle') # px.timelineの引数textを置く位置を内側の中央に変更

        # ラベルを手動で配置するためのannotationsを作成
        annotations = []
        prio = {'午前': 2, '当日': 1, '　　': 0}
        for i in range(len(data["仕事名"])):
            # 作業IDを棒の左側に配置
            if data["前後"][i] == "Before":
                annotation_work_id = dict(
                    x=data["開始時刻"][i] + timedelta(minutes=-7),
                    y=data["順番"][i],
                    text=str(data["ID"][i]) + '*' * prio[data['優先'][i]],
                    showarrow=False
                )
                annotations.append(annotation_work_id)

            # 作業時間を棒の中央に配置
            annotation_work_time = dict(
                x=data["開始時刻"][i] + (data["終了時刻"][i] - data["開始時刻"][i]) / 2,
                y=data["順番"][i],
                text=str((data["終了時刻"][i] - data["開始時刻"][i]).seconds // 60),
                showarrow=False,
                font=dict(size=10)
            )
            annotations.append(annotation_work_time)
        fig.update_layout(annotations=annotations)  # annotationsを設定

        # 昼休みなどの時間に縦線を付ける
        max_y = len(set(data["ID"])) + 0.5
        for t in TT:
            fig.add_shape(
                dict(
                    type="line",
                    x0=add_minutes_to_datetime(t),
                    x1=add_minutes_to_datetime(t),
                    y0=0.5,
                    y1=max_y,
                    line=dict(color="red", width=1)
                )
            )
        # fig.show()            # ipynbでの実行時はこちら
        st.plotly_chart(fig, theme=None)  # theme=None: デザインをstreamlit版にしない

    # ガントチャート描画
    st.write('最適なスケジュールを立案しました．')
    draw_schedule(df_schedule)

    # # 作成した仕事の表示
    # d = df.query('x + y + z > 0')
    # print('## 生産した仕事 ', len(d), '個')
    # print(d.loc[:, :'セット数'])

    # # 作成できなかった作業の表示
    # d = df.query('x + y + z == 0')
    # print('\n ## 生産しなかった仕事 ', len(d), '個')
    # print(d.loc[:, :'セット数'])

    # 出力用DataFrameの作成
    df_out = df_schedule.copy()
    df_out.drop(columns=['仕事名', '順番', '前後', '優先'], inplace=True)
    df_min = df_out.groupby('ID')['開始時刻'].min()
    df_max = df_out.groupby('ID')['終了時刻'].max()
    df_out = pd.concat([df_min, df_max], axis=1)
    df_out.columns = ['開始時刻', '終了時刻']
    df_out['開始時刻'] = df_out['開始時刻'].apply(lambda x: x.strftime('%H:%M'))
    df_out['終了時刻'] = df_out['終了時刻'].apply(lambda x: x.strftime('%H:%M'))
    df_out = pd.merge(df_opt, df_out, on='ID', how='left')
    df_out.drop(columns=['x', 'v', 'y', 'w', 'z'], inplace=True)
    return df_out


def execute_optimization(df):
    # 初期設定
    preparation()

    # 最適化の実行
    if (df_opt := solve_model1(df)) is None:
        return None, None

    # 最適化の結果に基づいたスケジュールの立案
    df_schedule = construct_schedule(df_opt)

    return df_opt, df_schedule
