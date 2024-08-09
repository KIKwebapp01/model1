import streamlit as st


def change_settings():
    st.title('設定変更')


def main():
    st.sidebar.markdown(
        """
        ## スケジュール自動作成アプリ
        """
    )

    page = st.sidebar.selectbox('アクションを選択してください', ['立案', '設定変更'])
    if page == '立案':
        st.title('スケジュール作成')
    elif page == '設定変更':
        change_settings()
    else:
        st.title('')


main()