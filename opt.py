from mip import *
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import pandas as pd

# TT = [0, 210, 260, 420, 430, 530]  # 各時刻．AM開始，AM終了，PM1開始，PM1終了，PM2開始，PM2終了
# T = dict(zip([1, 12, 2, 23, 3], [TT[i + 1] - TT[i] for i in range(5)]))  # 各時間．AM，昼休み，PM1，PM休み，PM2


### 前準備
def preparation():
    global TT, T
    tt = st.session_state.tt['時刻']
    today = datetime.now().date()
    TT = [(datetime.combine(today, tt[i]) - datetime.combine(today, tt[0])).seconds // 60 for i in range(6)]
    T = dict(zip([1, 12, 2, 23, 3], [TT[i + 1] - TT[i] for i in range(5)]))

# 変数の値（整数）を返す．実数で格納されているため，誤差対策
def val(variable):
    return int(variable.x + 0.1)


def solve_base_model(df, determined = dict()):
    # #定数用のデータの作成
    J = df.index
    a, bb, n = df['自動前段取'], df['自動加工'], df['セット数']
    b = {j: bb[j] / n[j] for j in J}
    prioAM, prioToday = df['午前優先'], df['当日優先']
    alpha = [10000, 1000, 0.001]  # 重み．0:午前優先，1:当日優先，2:早く終わる価値
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

    # 仕事そのものの条件
    for j in J:
        model += x[j] + y[j] + z[j] <= 1  # ①：始めるならいずれかの時間帯
    model += xsum(v[j] for j in J) <= 1  # ②：昼休みにできる仕事（v[j] = 1）は1個まで
    for j in J:
        model += v[j] <= x[j]  # ③：昼休みにできる仕事は，AMに始める
    model += xsum(w[j] for j in J) <= 1  # ④：午後休みにできる仕事（w[j] = 1）は1個まで
    for j in J:
        model += 2 * w[j] <= 2 * y[j] + v[j] + xi  # ⑤：午後休みにできる仕事は，午後1に始めた仕事か昼休みから続く（v[j] = 1）仕事

    # #昼休み終了まで
    model += xsum(a[j] * x[j] + n[j] * b[j] * (x[j] - v[j]) for j in J) <= T[1]  # ⑥：AM中に終わらせる仕事
    model += t1 <= xsum(a[j] * x[j] + n[j] * b[j] * (x[j] - v[j]) + b[j] * v[j] for j in J) - T[1] * xsum(
        v[j] for j in J)  # ⑦：昼休み作業時間の上限
    model += t1 <= T[12] * xsum(v[j] for j in J)  # ⑧：昼休み作業時間の上限

    # 午前最後の仕事が，午後1でも終わらない場合の処理．ξ=1となる．
    model += xsum((a[j] + b[j] * n[j]) * x[j] for j in J) <= T[1] + t1 + T[2] + T[3] * xi  # ⑨：ξフラグ
    for j in J:
        model += y[j] <= 1 - xi  # ⑩：ξ=1のときは，午後1に仕事を始めない
        model += w[j] <= 1 - xi + v[j]  # ⑪：ただし，v=1の仕事を午後休みに行うのはOK

    # #午後1終了まで
    model += xsum(a[j] * (x[j] + y[j]) + n[j] * b[j] * (x[j] + y[j] - w[j]) for j in J) <= T[1] + t1 + T[
        2]  # ⑫：午後1までに終わらせる
    model += t1 + t2 <= xsum(a[j] * (x[j] + y[j]) + n[j] * b[j] * (x[j] + y[j] - w[j]) + b[j] * w[j] for j in J) - (
                T[1] + T[2]) * xsum(w[j] for j in J)
    # ⑬：午後休み作業時間の上限
    model += t2 <= T[23] * xsum(w[j] for j in J)  # ⑭：午後休み作業時間の上限

    # #終了まで
    # model += xsum((a[j] + n[j] * b[j]) * z[j] for j in J) <= T[3]       # ⑯：午後2で始めた仕事は全部終わらせる
    model += xsum((a[j] + n[j] * b[j]) * (x[j] + y[j] + z[j]) for j in J) <= T[1] + t1 + T[2] + t2 + T[3]  # ⑮：全部終わらせる

    # 値が決定している変数の設定
    for j, var_set in determined.items():
        if 'x' in var_set:      # x[j] = 1 の場合
            model += x[j] == 1
            if 'v' in var_set:  # v[j] = 1の場合
                model += v[j] == 1
            else:
                model += v[j] == 0
            if 'w' in var_set:  # w[j] = 1の場合
                model += w[j] == 1
            else:
                model += w[j] == 0
        elif 'y' in var_set:    # y[j] = 1の場合
            model += y[j] == 1
            if 'w' in var_set:  # w[j] = 1の場合
                model += w[j] == 1
            else:
                model += w[j] == 0
        elif 'z' in var_set:    # z[j] = 1の場合
            model += z[j] == 1

    # #目的関数の設定
    model.objective = (maximize(alpha[0] * xsum(prioAM[j] * x[j] for j in J)
                                + alpha[1] * xsum(prioToday[j] * (3 * x[j] + 2 * y[j] + z[j]) for j in J)
                                + xsum(3 * x[j] + 2 * y[j] + z[j] for j in J)
                                - alpha[2] * (t1 + t2 + xsum(v[j] + w[j] for j in J)))
                       )
    model.write("model.lp")

    # #最適化の実行
    model.verbose = 0  # 実行過程の非表示
    status = model.optimize()

    # #最適化の結果出力
    if status == OptimizationStatus.OPTIMAL:
        df_ret = df.copy()
        df_ret.loc[J, 'x'] = [val(x[j]) for j in J]
        df_ret.loc[J, 'v'] = [val(v[j]) for j in J]
        df_ret.loc[J, 'y'] = [val(y[j]) for j in J]
        df_ret.loc[J, 'w'] = [val(w[j]) for j in J]
        df_ret.loc[J, 'z'] = [val(z[j]) for j in J]
        # print(df)
        # print(f'{val(t1) = }, {val(t2) = }, {val(xi) = }')
        return df_ret
    else:
        # print('最適解が求まりませんでした。')
        return None

##最適化
def solve_model1(df):
    return solve_base_model(df)


def solve_model2(df):
    cvt_df = df.copy()
    cvt_df['納期'] = df['納期'].astype(str)       # 納期情報をdatetime.dateからstrに変換
    due_list = sorted(set(cvt_df['納期'].tolist()))
    determined = dict()     # 値が決定した作業の情報
    df_opt = None
    for i in range(len(due_list)):
        target_dues = due_list[:i+1]                # 対象とする納期
        cur_df = cvt_df[cvt_df['納期'].isin(target_dues)]    # 対象とするデータ（行）
        df_opt = solve_base_model(cur_df, determined)
        if df_opt is None:
            break
        determined = {j : set() for j in df_opt.index}      # determined[j]: 仕事jで値が1になった変数．value: set型
        for col in ['x', 'y', 'z', 'v', 'w']:
            matching_indices = df_opt.index[df_opt[col] == 1].tolist()
            for j in matching_indices:
                determined[j].add(col)
    return df_opt


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

    # print("作成仕事数：", (df['x'] + df['y'] + df['z']).sum())
    # print("作成数量　：", ((df['x'] + df['y'] + df['z']) * df['セット数']).sum())

    #仕事一覧とその仕事の開始時刻、終了時刻
    result = {"仕事名": [], "ID": [], "開始時刻": [], "終了時刻": [], "順番": [], "前後": [], "優先": []}
    # d = df.sort_values(by=['午前優先', '当日優先'], ascending=False)
    d = df.sort_values(by=['納期', "ID"])
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
    df_out.sort_values(by=['納期', "ID"], inplace=True)
    return df_out


def execute_optimization(df):
    # 初期設定
    preparation()

    # 最適化の実行
    df_opts = [solve_model1(df), solve_model2(df)]
    if any(df_opt is None for df_opt in df_opts):     # どちらかのモデルが解けなかったら
        return (None, None), (None, None)

    # 最適化の結果に基づいたスケジュールの立案
    df_schedules = [construct_schedule(df_opt) for df_opt in df_opts]

    return df_opts, df_schedules
