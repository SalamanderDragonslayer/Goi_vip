import streamlit as st
# Đường link đến trang Thần số học, Sinh trắc học, Nhân tướng học
thanosohoc_link_vip = "https://directionalpathway-thansohoc-vip.streamlit.app/"
sinhtrachoc_link_vip = "https://directionalpathway-sinhtrachoc-vip.streamlit.app/"
nhantuonghoc_link_vip = "https://directionalpathway-nhantuonghoc-vip.streamlit.app/"
# Cài đặt trang để hiển thị layout "wide"
st.set_page_config(layout="wide")
col1, col2 = st.columns([12, 1])

with col1:
    st.markdown('<a href="{}" style="color: white; text-decoration: none;"><button style="background-color: #DD83E0; border: none; border-radius: 5px; padding: 10px 20px; cursor: pointer;">Sign In</button></a>'.format(thanosohoc_link), unsafe_allow_html=True)

# Đặt nút "Sign Up" trong cột thứ hai
with col2:
    st.markdown('<a href="{}" style="color: white; text-decoration: none;"><button style="background-color: #DD83E0; border: none; border-radius: 5px; padding: 10px 20px; cursor: pointer;">Sign Up</button></a>'.format(thanosohoc_link), unsafe_allow_html=True)
# Tiêu đề của trang
st.title("Chào mừng đến với Direction-Pathway")
st.write(
    "Chúng ta hãy khám phá những khía cạnh thú vị về vận mệnh và tính cách của bạn thông qua thần số học, sinh trắc học và nhân tướng học")



# Tiêu đề "Thần số học" sẽ là một hyperlink
st.title("Chào mừng đến với Trang chủ Thần số học, Sinh trắc học vân tay và Nhân tướng học")
st.write("Chúng ta hãy khám phá những khía cạnh thú vị về vận mệnh, tính cách và vân tay của bạn!")

# Chuyển trang tới đường link "Thần số học" khi tiêu đề được nhấp
st.markdown('<h2 style="color: #DD83E0;"><a style="color: #DD83E0;text-decoration: none;" href="{}">Thần số học</a></h2>'.format(thanosohoc_link), unsafe_allow_html=True)
st.write(
    "Thần số học là nghệ thuật dựa trên việc phân tích các số liên quan đến ngày, tháng và năm sinh của bạn để hiểu về vận mệnh và tính cách.")

# Chuyển trang tới đường link "Sinh trắc học" khi tiêu đề được nhấp
st.markdown('<h2 style="color: #DD83E0;"><a style="color: #DD83E0;text-decoration: none;" href="{}">Sinh trắc học</a></h2>'.format(sinhtrachoc_link), unsafe_allow_html=True)
st.write(
    "Sinh trắc học vân tay là nghiên cứu về các đặc điểm vân tay để xác định tính cách và tương lai của một người.")

# Chuyển trang tới đường link "Nhân tướng học" khi tiêu đề được nhấp
st.markdown('<h2 style="color: #DD83E0;"><a style="color: #DD83E0;text-decoration: none;" href="{}">Nhân tướng học</a></h2>'.format(nhantuonghoc_link), unsafe_allow_html=True)
st.write(
    "Nhân tướng học là nghiên cứu về các đặc điểm gương mặt để xác định tính cách và tương lai của một người.")


