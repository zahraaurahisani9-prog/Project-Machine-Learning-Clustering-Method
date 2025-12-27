import streamlit as st


# ==================================================
# UI CARD HELPER
# ==================================================
def card(title, description=""):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# CONTACT PAGE
# ==================================================
def contact_me():

    card(
        "📬 Kontak & Kolaborasi",
        "Jika Anda memiliki pertanyaan, masukan, atau ingin berdiskusi "
        "lebih lanjut terkait aplikasi ini, silakan hubungi saya melalui "
        "kanal berikut."
    )

    st.markdown("""
    **Email**  
    zahraaurahisani9@gmail.com  

    **LinkedIn**  
    https://www.linkedin.com/in/zahra-aura-hisani  

    **GitHub**  
    https://github.com/zahraaurahisani9-prog  
    """)

    card(
        "🤝 Kolaborasi",
        "Saya terbuka untuk diskusi dan kolaborasi dalam bidang "
        "data science, machine learning, dan analisis data."
    )

    st.success(
        "Terima kasih telah menggunakan Clustering Analysis Platform."
    )
