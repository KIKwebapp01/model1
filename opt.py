from mip import Model, xsum, maximize, OptimizationStatus
import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

global alpha
global I_A, I_M1, I_M2, I_M, I
global J, J_group
global M, T, T_, ST, TT, START_TIME
global a, b, N, e, p

# 時間経過後の時刻を返す関数
def add_minutes_to_datetime(minute_to_add):
    # 指定された日時をdatetimeオブジェクトに変換
    now = datetime.datetime.now()
    dt = pd.to_datetime(f"{now.year}-{now.month}-{now.day}-{START_TIME}", format='%Y-%m-%d-%H-%M')
    # 指定された分の時間を加算
    return dt + datetime.timedelta(minutes = float(minute_to_add))

# ## 最適化を実行して，最適なスケジュールを求める（ガントチャート表示用）
def execute_optimization(df_data, df_time):
    # ### 前準備
    def preparation(df_data, df_time):
        # ## 時刻・時間の設定
        def set_periods(df_time):
            global J, J_group
            global M, T, T_, ST, TT, START_TIME

            tt = df_time['時刻'].apply(
                lambda t: datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second))  # datedelta型に変換
            start_time = tt.iloc[0]  # 始業時刻．Timedelta型
            base_times = tt.apply(lambda t: (t - start_time).total_seconds() // 60).astype(int).tolist()  # 経過分数をリストに変換

            M = len(base_times) // 2  # スロット数
            T = list(range(1, M + 1))  # [1, 2, ..., M]
            T_ = T[:-1]  # [1, 2, ..., M-1]
            ST, TT = {}, {}
            for i in range(M):
                ST[i + 1] = base_times[i * 2]
                ST[i + 1, i + 2] = base_times[i * 2 + 1]
                TT[i + 1] = base_times[i * 2 + 1] - base_times[i * 2]
                if i < M - 1:
                    TT[i + 1, i + 2] = base_times[i * 2 + 2] - base_times[i * 2 + 1]
            # ST = {1:0, (1,2):210, 2:260, (2,3):420, 3:430, (3,4):530}
            # TT = {1:210, (1,2):50, 2:160, (2,3):10, 3:100}
            START_TIME = f'{start_time.components.hours}-{start_time.components.minutes}'  # Timedelta型から時間と分を抽出

        # ## 仕事関連の定数データの設定
        def set_constant(df_data):
            global a, b, N, e, p

            a, b, N, e, p = {}, {}, {}, {}, {}
            for row in df_data.itertuples():
                j = row.Index  # 行インデックス＝仕事ID
                N[j] = row.セット数
                if row.得意先 == 1:
                    if pd.notna(row.自動前段取) and pd.notna(row.自動加工):  # 自動にデータが入力されているなら
                        for i in I_A:  # 自動担当
                            a[i, j] = row.自動前段取
                            b[i, j] = row.自動加工 // N[j]
                            e[i, j] = 1
                    if pd.notna(row.大隅前段取) and pd.notna(row.大隅加工):  # 大隅にデータが入力されているなら
                        for i in I_M1:  # 手動担当取引先1
                            a[i, j] = row.大隅前段取
                            b[i, j] = row.大隅加工 // N[j]
                            e[i, j] = 1
                else:  # 得意先 == 2
                    if pd.notna(row.大隅前段取) and pd.notna(row.大隅加工):  # 大隅にデータが入力されているなら
                        for i in I_M2:  # 手動担当取引先2
                            a[i, j] = row.大隅前段取
                            b[i, j] = row.大隅加工 // N[j]
                            e[i, j] = 1
            prioAM, prioToday = df_data['午前優先'], df_data['当日優先']
            for t in T:
                for j in J:
                    if prioAM[j] == 1:
                        p[j, t] = alpha[0] * (M - t + 1)
                    elif prioToday[j] == 1:
                        p[j, t] = alpha[1] * (M - t + 1)
                    else:
                        p[j, t] = M - t + 1

        global alpha
        global I_A, I_M1, I_M2, I_M, I
        global J, J_group

        # 最適化で用いる定数の初期化
        alpha = [10000, 100]  # 好ましさの重み．午前優先，当日優先
        I_A, I_M1, I_M2 = [1], [2], [3]  # 自動機械担当，手動取引先1担当，（手動）取引先2担当
        I_M = I_M1 + I_M2  # 手動担当
        I = I_A + I_M  # 全作業員
        J = df_data.index  # 仕事IDのリスト
        J_group = [set(group.index) for _, group in df_data.groupby('納期', sort=True)]  # 納期順に，仕事IDをグループ化する
        set_periods(df_time)  # スロット数，スロット番号，スロット番号（最後除く），スロット開始時刻，スロット期間，始業時刻
        set_constant(df_data)  # 仕事に関する定数

    ## 最適化を実行
    def solve_model(mode):
        def solve_model_core(cur_J, determined):  # cur_J: 割当対象仕事ID集合，determined: 過去の割当結果
            def val(variable):
                # 変数の値（整数）を返す．実数で格納されているため，誤差対策
                return int(variable.x + 0.1)

            # 空問題の作成
            model = Model('Schedule')

            # #決定変数の作成
            x, y, z, s, n, xi, tau = {}, {}, {}, {}, {}, {}, {}
            for i, j in e:  # ①：作業員は担当不可の仕事を担当しない
                if j not in cur_J:
                    continue  # 仕事jが今回の対象外ならスキップ
                x[i, j] = model.add_var(f'x{i}_{j}', var_type='B')  # 作業員iが仕事jを行うか（0/1)
                for t in T:
                    y[i, j, t] = model.add_var(f'y{i}_{j}_{t}', var_type='B')  # 作業員iが仕事jの加工をスロットtで行うか(0/1)
                    z[i, j, t] = model.add_var(f'z{i}_{j}_{t}', var_type='B')  # 作業員iが行う仕事jの加工がスロットtを超えるか(0/1)
                    s[i, j, t] = model.add_var(f's{i}_{j}_{t}', var_type='B')  # 作業員iが仕事jの前段取をスロットtで行うか(0/1)
                    n[i, j, t] = model.add_var(f'n{i}_{j}_{t}', var_type='I', lb=0)  # 作業員iが仕事jをスロットtで開始するセット数（自然数）
            for i in I:
                for t in T:
                    xi[i, t] = model.add_var(f'xi{i}_{t}',
                                             var_type='B')  # 作業員iが(休みtから休みt+1まで同一の仕事が占有して）スロットtで仕事を開始できないか(0/1)
                    tau[i, t] = model.add_var(f'tau{i}_{t}', lb=0)  # 作業員iがスロットt以降で処理すべき残作業時間

            # #制約条件の追加
            for j in cur_J:
                model += xsum(x[i, j] for i in I if (i, j) in x) <= 1  # ②：作業は分担しない
            for i, j in x:
                model += xsum(n[i, j, t] for t in T) == N[j] * x[i, j]  # ③：作業員iが仕事jを行うなら，合計N_jセット生産する
            for i, j, t in y:
                model += y[i, j, t] <= x[i, j]  # ④：作業員iが仕事jを行なわないなら，どのスロットでも生産できない
            for i, j, t in y:
                model += y[i, j, t] <= n[i, j, t]
                model += n[i, j, t] <= N[j] * y[i, j, t]  # ⑤：作業員iが仕事jをスロットtで生産するなら1つ以上生産し，行わないなら1つも生産しない
            for i, j in x:
                model += xsum(s[i, j, t] for t in T) <= x[i, j]  # 〇：作業員iは担当しない仕事jの段取を行わない
            for i, j, t in y:
                model += s[i, j, t] <= y[i, j, t]  # ⑥：作業員iがスロットtで段取したら，スロットtで作業も行う
            for i, j in x:
                model += y[i, j, 1] <= s[i, j, 1]  # ⑦：作業員iがスロット1で仕事をするには，段取を行わなければならない
            for i, j in x:
                for t in T_:
                    model += y[i, j, t + 1] <= s[i, j, t + 1] + z[
                        i, j, t]  # ⑧：作業員iがスロットt+1でできる仕事jは，スロットt+1で段取を行う仕事か，スロットtから継続する仕事である
            for i in I:
                for t in T:
                    model += xsum(z[i, j, t] for j in cur_J if (i, j, t) in z) <= 1  # ⑨：作業員iがスロットtを超えて行うことができる仕事は1つまで
            for i, j in x:
                for t in T:
                    model += z[i, j, t] <= y[i, j, t]  # ⑩：作業員iがスロットtを超えて行う仕事は，スロットtで行っていた仕事のみ
            for i, j in x:
                for t in T_:
                    model += y[i, j, t] + y[i, j, t + 1] <= 1 + z[i, j, t]  # ⑪：連続スロットt,t+1で続けて仕事を行うには，休憩tで仕事を行う必要がある
            for i, j in x:
                for t in T_:
                    model += z[i, j, t] + z[i, j, t + 1] <= 1 + xi[
                        i, t + 1]  # ⑫：スロットt+1の前後で作業員iが同じ仕事をするならξ_{i,t+1}=1とする．
            for i, j in x:
                for t in T:
                    model += s[i, j, t] + xi[i, t] <= 1  # ⑬：ξ_{i,t}=1なら，作業員iはスロットtで段取できない
            for i, t in tau:  # ⑭：作業員iは，継続作業1セット分を除いて，スロット時間内で仕事を終わらせる
                model += tau[i, t] + xsum(
                    a[i, j] * s[i, j, t] + b[i, j] * (n[i, j, t] - z[i, j, t]) for j in cur_J if (i, j) in x) <= TT[t]
            for i in I_A:
                for t in T_:  # ⑮：自動機械担当作業員iの作業は，スロット時間を超えた分は，続く休みとそれ以降のスロットで行う
                    model += tau[i, t] + xsum(
                        a[i, j] * s[i, j, t] + b[i, j] * n[i, j, t] for j in cur_J if (i, j) in x) <= \
                             TT[t] + TT[t, t + 1] + tau[i, t + 1]
            for i in I_M:
                for t in T_:  # ⑯：手動機械担当作業員iの作業は，スロット時間を超えた分は，以降のスロットで行う
                    model += tau[i, t] + xsum(
                        a[i, j] * s[i, j, t] + b[i, j] * n[i, j, t] for j in cur_J if (i, j) in x) <= \
                             TT[t] + tau[i, t + 1]
            # ⑰⑱⑲：境界条件
            for i, j in x:
                model += z[i, j, M] == 0
            for i in I:
                model += xi[i, 1] == 0
                model += tau[i, 1] == 0
            # 過去の計算結果の継続
            for k, v in determined.items():
                if k[0] == 'x':
                    model += x[k[1], k[2]] == v
                elif k[0] == 'y':
                    model += y[k[1], k[2], k[3]] == v
                elif k[0] == 'z':
                    model += z[k[1], k[2], k[3]] == v
                elif k[0] == 's':
                    model += s[k[1], k[2], k[3]] == v

            # #目的関数の設定
            model.objective = maximize(xsum(p[j, t] * n[i, j, t] for (i, j, t) in n))
            # model.write("model.lp")

            # #最適化の実行
            status = model.optimize()

            # #最適化の結果出力
            if status == OptimizationStatus.OPTIMAL:
                dfs = {i: dict() for i in I}
                for i in I:
                    dfs[i]['ID'] = [j for j in cur_J if (i, j) in x]
                    dfs[i]['x'] = [val(x[i, j]) for j in cur_J if (i, j) in x]
                    for t in T:
                        dfs[i][f's{t}'] = [val(s[i, j, t]) for j in cur_J if (i, j) in x]
                        dfs[i][f'n{t}'] = [val(n[i, j, t]) for j in cur_J if (i, j) in x]
                        dfs[i][f'y{t}'] = [val(y[i, j, t]) for j in cur_J if (i, j) in x]
                        dfs[i][f'z{t}'] = [val(z[i, j, t]) for j in cur_J if (i, j) in x]
                    dfs[i] = pd.DataFrame(dfs[i]).set_index('ID')  # 辞書→データフレーム
                    # display(dfs[i])
                    # print('τ = ', end='')
                    # print(*[val(tau[i,t]) for t in T])
                    # print('ξ= ', end='')
                    # print(*[val(xi[i,t]) for t in T])
                cur_determined = dict()
                for (i, j) in x:
                    cur_determined['x', i, j] = val(x[i, j])
                    for t in T:
                        cur_determined['y', i, j, t] = val(y[i, j, t])
                        cur_determined['z', i, j, t] = val(z[i, j, t])
                        cur_determined['s', i, j, t] = val(s[i, j, t])
            else:
                print('最適解が求まりませんでした。')
                dfs, cur_determined = None, None
            return dfs, cur_determined  # dfs: 作業員ごとの割当結果，cur_determined: 最適解

        dfs = None  # 初期化
        if mode == 1:
            dfs, determined = solve_model_core(set(J), dict())  # 完成数量の最大化を1回のみ実行
        elif mode == 2:
            cur_J, determined = set(), dict()
            for gr in J_group:
                cur_J |= set(gr)  # 今回対象となる仕事IDリストを追加
                dfs, determined = solve_model_core(cur_J, determined)  # dfs: 作業員ごとの割当結果(df). キーは作業員
                if dfs is None:
                    break  # 解がないならbreakしてNoneを返す
        if dfs is None:  # 解なし
            return None
        return dfs  # 各作業員の最適化の結果．キーが作業員

    ### 最適化の結果から，生産時間を算出
    def construct_schedule(dfs_sol, df_data):   # dfs_sol: 各作業員の最適化結果（モデル情報はない）
        # 1つの仕事の開始・終了時刻を計算し，resultに書き込み，cur_timeを更新
        # row: 行データ，t：作業時間，ty：タイプ（Before/After）, order：通し番号
        def write_job(result, row, i, cur_time, t, ty, order):
            result["作業員"].append(i)
            result["仕事名"].append(f"Job{row.Index}_{ty}")
            result["ID"].append(row.Index)
            result["開始時刻"].append(add_minutes_to_datetime(cur_time))
            result["終了時刻"].append(add_minutes_to_datetime(cur_time + t))
            result["順番"].append(order)
            result["前後"].append(ty)
            if row.午前優先:
                result["優先"].append('午前')
            elif row.当日優先:
                result["優先"].append('当日')
            else:
                result["優先"].append(pd.NA)      # NaNを格納
            return cur_time + t

        # 最適化の結果を自動機械作業員用スケジュールに変換
        def construct_op_auto_schedule(df, i):
            result = {"作業員": [], "仕事名": [], "ID": [], "開始時刻": [], "終了時刻": [], "順番": [], "前後": [],
                      "優先": []}
            d = df.sort_values(by=['午前優先', '当日優先'], ascending=False)
            order = 1
            cur_time = 0

            # AM1で終わる仕事
            for row in d.itertuples():
                if row.y1 > 0 and row.z1 == 0:
                    cur_time = write_job(result, row, i, cur_time, row.自動前段取, 'Before', order)
                    for _ in range(row.n1):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    order += 1
            # AM1で開始するが終わらない仕事
            for row in d.itertuples():
                if row.z1 > 0:
                    # AM1と昼休みまでの仕事
                    cur_time = write_job(result, row, i, cur_time, row.自動前段取, 'Before', order)
                    for _ in range(row.n1):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    # 昼休みで終わる場合
                    if row.y2 == 0:
                        order += 1
                        break  # PM1の処理へ
                    # PM1も続く場合
                    cur_time = max(cur_time, ST[2])  # PM1から開始する時刻
                    for _ in range(row.n2):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    # PM1で終わる場合
                    if row.y3 == 0:
                        order += 1
                        break  # PM1の処理へ
                    # 午後休みも続く場合
                    cur_time = max(cur_time, ST[3])  # PM2から開始する時刻
                    for _ in range(row.n3):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    order += 1
                    break

            # PM1内で開始して終わる仕事
            cur_time = max(cur_time, ST[2])  # PM1から開始する時刻
            for row in d.itertuples():
                if row.z1 == 0 and row.y2 > 0 and row.z2 == 0:
                    cur_time = write_job(result, row, i, cur_time, row.自動前段取, 'Before', order)
                    for _ in range(row.n2):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    order += 1

            # PM1で開始するが，PM1で終わらない仕事
            for row in d.itertuples():
                if row.z1 == 0 and row.y2 > 0 and row.z2 > 0:
                    cur_time = write_job(result, row, i, cur_time, row.自動前段取, 'Before', order)
                    for _ in range(row.n2):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    # PM休みで終わる場合
                    if row.y3 == 0:
                        order += 1
                        break  # PM2の処理へ
                    # PM2も続く場合
                    cur_time = max(cur_time, ST[3])  # PM2から開始する時刻
                    for _ in range(row.n3):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    order += 1
                    break

            # PM2で開始して終わる仕事
            cur_time = max(cur_time, ST[3])  # PM2から開始する時刻
            for row in d.itertuples():
                if row.z2 == 0 and row.y3 > 0:
                    cur_time = write_job(result, row, i, cur_time, row.自動前段取, 'Before', order)
                    for _ in range(row.n3):
                        cur_time = write_job(result, row, i, cur_time, row.自動加工 // row.セット数, 'After', order)
                    order += 1

            return pd.DataFrame(result)

        # 最適化の結果を手動機械作業員用スケジュールに変換
        def construct_op_manual_schedule(df, i):
            result = {"作業員": [], "仕事名": [], "ID": [], "開始時刻": [], "終了時刻": [], "順番": [], "前後": [],
                      "優先": []}
            d = df.sort_values(by=['午前優先', '当日優先'], ascending=False)
            order = 1
            cur_time = 0

            # AM1で終わる仕事
            for row in d.itertuples():
                if row.y1 > 0 and row.z1 == 0:
                    cur_time = write_job(result, row, i, cur_time, row.大隅前段取, 'Before', order)
                    for _ in range(row.n1):
                        cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                    order += 1
            # AMで開始するが終わらない仕事
            for row in d.itertuples():
                if row.z1 > 0:
                    # AMまでの仕事
                    cur_time = write_job(result, row, i, cur_time, row.大隅前段取, 'Before', order)
                    for _ in range(row.n1 - 1):  # AMで入る分のセット
                        cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                    left_time = ST[1, 2] - cur_time  # AMの残り時間
                    cur_time = write_job(result, row, i, cur_time, left_time, 'After', order)  # 昼休みまで作業
                    left_time = row.大隅加工 // row.セット数 - left_time  # 残り作業時間
                    # PM1からの仕事
                    cur_time = ST[2]  # PM1から開始
                    if row.y3 == 0:  # PM1で終わる場合
                        cur_time = write_job(result, row, i, cur_time, left_time, 'After', order)
                        for _ in range(row.n2):
                            cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                        order += 1
                        break
                    # PM2へも続く場合
                    left_time = ST[2, 3] - cur_time  # PM1の残り時間
                    cur_time = write_job(result, row, i, cur_time, left_time, 'After', order)  # 午後休みまで作業
                    left_time = row.大隅加工 // row.セット数 - left_time  # 残り作業時間
                    # PM2からの仕事
                    cur_time = ST[3]  # PM2から開始
                    cur_time = write_job(result, row, i, cur_time, left_time, 'After', order)
                    for _ in range(row.n3):
                        cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                    order += 1
                    break

            # PM1内で開始して終わる仕事
            cur_time = max(cur_time, ST[2])  # PM1から開始する時刻
            for row in d.itertuples():
                if row.z1 == 0 and row.y2 > 0 and row.z2 == 0:
                    cur_time = write_job(result, row, i, cur_time, row.大隅前段取, 'Before', order)
                    for _ in range(row.n2):
                        cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                    order += 1

            # PM1で開始するが，PM1で終わらない仕事
            for row in d.itertuples():
                if row.z1 == 0 and row.y2 > 0 and row.z2 > 0:
                    cur_time = write_job(result, row, i, cur_time, row.大隅前段取, 'Before', order)
                    for _ in range(row.n2 - 1):
                        cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                    left_time = ST[2, 3] - cur_time  # PM1の残り時間
                    cur_time = write_job(result, row, i, cur_time, left_time, 'After', order)  # 午後休みまで作業
                    left_time = row.大隅加工 // row.セット数 - left_time  # 残り作業時間
                    # PM2からの仕事
                    cur_time = ST[3]  # PM2から開始
                    cur_time = write_job(result, row, i, cur_time, left_time, 'After', order)
                    for _ in range(row.n3):
                        cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                    order += 1
                    break

            # PM2で開始して終わる仕事
            cur_time = max(cur_time, ST[3])  # PM2から開始する時刻
            for row in d.itertuples():
                if row.z2 == 0 and row.y3 > 0:
                    cur_time = write_job(result, row, i, cur_time, row.大隅前段取, 'Before', order)
                    for _ in range(row.n3):
                        cur_time = write_job(result, row, i, cur_time, row.大隅加工 // row.セット数, 'After', order)
                    order += 1

            return pd.DataFrame(result)

        dfs_schedule = dict()
        for i, df_sol in dfs_sol.items():   # 各作業員の最適化の結果
            df_cur = pd.merge(df_sol, df_data, on='ID', how='left')  # 最適化の結果と仕事情報を結合
            if i in I_A:
                df_schedule = construct_op_auto_schedule(df_cur, i)  # 自動機械担当作業員のスケジュール（作業分割考慮）
            else:
                df_schedule = construct_op_manual_schedule(df_cur, i)  # 手動機械担当作業員のスケジュール（作業分割考慮）
            dfs_schedule[i] = df_schedule     # 変換後のスケジュールを辞書に格納
        return dfs_schedule     # 各作業員のスケジュール（モデル情報はない）

    # 初期設定
    preparation(df_data, df_time)

    # 最適化の実行
    dfs_schedule = dict()
    for mode in [1, 2]:
        dfs_sol = solve_model(mode)  # 各モデルでの最適化を実行．作業員ごとの割当結果(df)を得る．キー:作業員
        if dfs_sol is None:
            dfs_schedule = None     # 最適解が見つからなければ，Noneを返す
            break
        # ガントチャート用のスケジュール(df)に変換．セット分割や休憩分割対応
        dfs_schedule[mode] = construct_schedule(dfs_sol, df_data)   # モデルmodeでの各作業員のスケジュール
    return dfs_schedule     # モデルごと，作業員ごとのスケジュール(df)．2次元辞書

# ## ガントチャートを作成する
def create_gantt_charts(dfs_schedule):  # dfs_schedule：作業員・モデルごとのdf
    # #ガントチャートの描画関数
    def create_gantt_chart(df):
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
        return fig

    gantt_charts = dict()
    for mode, dfs_schedule_mode in dfs_schedule.items():  # モデル順に実行
        gantt_charts_mode = dict()
        for i, df_schedule in dfs_schedule_mode.items():     # 各作業員のスケジュール
            gantt_charts_mode[i] = create_gantt_chart(df_schedule)  # ガントチャートの作成
        gantt_charts[mode] = gantt_charts_mode
    return gantt_charts     # モデルごと，作業員ごとのガントチャート

# ## 最適なスケジュールを表形式に要約する
def summarize_schedule(dfs_schedule, df_data):
    # 作業員・モデルごとのガントチャート用df（セット分割・休憩分割）-> 割当表dfに変換
    df_summarized = df_data.copy()
    for mode, dfs_schedule_mode in dfs_schedule.items():
        dfs_summarized = []
        for i, df_schedule in dfs_schedule_mode.items():
            df_cur = df_schedule.copy()
            df_cur.drop(columns=['仕事名', '順番', '前後', '優先'], inplace=True)    # 不要な列を削除
            df_min = df_cur.groupby('ID')['開始時刻'].min()     # 行:ID，列:開始時刻の最小値．1列
            df_max = df_cur.groupby('ID')['終了時刻'].max()     # 行:ID，列:終了時刻の最大値．1列
            df_cur = pd.concat([df_min, df_max], axis=1)       # 上の2つの列を横方向に結合
            df_cur.columns = ['開始', '終了']                   # 列名の変更
            df_cur['開始'] = df_cur['開始'].apply(lambda x: x.strftime('%H:%M'))    # 表示形式の変更
            df_cur['終了'] = df_cur['終了'].apply(lambda x: x.strftime('%H:%M'))    # 表示形式の変更
            df_cur.columns = [f'開始{mode}', f'終了{mode}']
            df_cur[f'担当{mode}'] = [f'作業員{i}'] * len(df_cur)
            dfs_summarized.append(df_cur)   # 各作業員の割当表をリストに追加
        df_summarized_mode = pd.concat(dfs_summarized)    # 各作業員の割当を縦方向に結合
        df_summarized = pd.merge(df_summarized, df_summarized_mode, on='ID', how='left')    # 仕事データに各モデルでの割付結果列を追加
    return df_summarized

# ## モデルごとの結果の比較表を作成する
def output_table(df_summarized, df_data):
    if len(df_summarized) == 0:
        return None
    dfs_pivot = []
    for mode in [1, 2]:
        df_pivot = pd.pivot_table(df_summarized, index=f'担当{mode}', values='セット数',
                                  aggfunc=['count', 'sum'], margins=True, margins_name='合計')
        df_pivot.columns = df_pivot.columns.droplevel(0)  # 最上位のマルチインデックスを削除
        df_pivot.columns = [f"数量{mode}", f"セット数{mode}"]
        dfs_pivot.append(df_pivot)

    # ピボットテーブルを統合
    df_pivot = pd.concat(dfs_pivot, axis=1).fillna(0).astype(int)
    return df_pivot
