import streamlit as st


# ==================================================
# UI CARD HELPER (HANYA TAMPILAN)
# ==================================================
def card(title):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# ABOUT APPLICATION (ISI TIDAK DIUBAH)
# ==================================================
def about_application():

    card("🔍 About This Application")
    st.markdown("""
    Aplikasi ini merupakan **platform analisis clustering (unsupervised learning)**
    yang dirancang untuk membantu pengguna **mengeksplorasi, membandingkan, dan
    menginterpretasikan struktur cluster pada data numerik** tanpa menggunakan
    label atau variabel target.

    Fokus utama aplikasi ini adalah **analisis eksploratif dan interpretatif**,
    bukan prediksi atau klasifikasi.
    """)

    card("🎯 Tujuan Pengembangan")
    st.markdown("""
    Aplikasi ini dikembangkan dengan tujuan untuk:

    - Menyediakan **alur analisis clustering end-to-end**, mulai dari eksplorasi
      data hingga interpretasi hasil clustering.
    - Memungkinkan pengguna **membandingkan berbagai algoritma clustering**
      pada dataset yang sama.
    - Membantu pengguna **memahami karakteristik dan perbedaan antar cluster**
      melalui analisis statistik dan deskriptif.

    Pendekatan ini menekankan **pemahaman struktur data** dibandingkan optimasi
    model atau akurasi prediktif.
    """)

    card("🧠 Algoritma Clustering")
    st.markdown("""
    Aplikasi ini mengimplementasikan **sepuluh algoritma clustering** dari berbagai
    pendekatan, yaitu:

    1. **K-Means** – Metode partisi berbasis centroid sebagai baseline clustering.
    2. **MiniBatch K-Means** – Varian K-Means yang lebih efisien untuk dataset besar.
    3. **Hierarchical Clustering (Agglomerative)** – Clustering bertingkat untuk
       memahami struktur hirarki data.
    4. **DBSCAN** – Metode berbasis kepadatan yang mampu mendeteksi noise dan cluster
       dengan bentuk arbitrer.
    5. **HDBSCAN** – Pengembangan DBSCAN dengan kepadatan adaptif (jika dependency tersedia).
    6. **OPTICS** – Metode density-based yang mengeksplorasi struktur cluster pada
       berbagai skala kepadatan.
    7. **Spectral Clustering** – Clustering berbasis graf untuk struktur data non-linear.
    8. **BIRCH** – Metode incremental yang efisien untuk data numerik skala besar.
    9. **Gaussian Mixture Model (EM)** – Pendekatan probabilistik berbasis distribusi Gaussian.
    10. **Discretization-based Clustering (Grid-inspired)** – Pendekatan clustering
        dengan mendiskretisasi ruang fitur sebelum proses pengelompokan.

    Keberagaman algoritma ini memungkinkan analisis yang lebih objektif terhadap
    karakteristik data yang berbeda.
    """)

    card("📊 Evaluasi Hasil Clustering")
    st.markdown("""
    Karena clustering tidak memiliki *ground truth*, kualitas hasil clustering
    dievaluasi menggunakan **metrik evaluasi internal**, yaitu:

    - **Silhouette Score** – Mengukur tingkat pemisahan antar cluster.
    - **Davies–Bouldin Index** – Mengukur kekompakan dan tumpang tindih antar cluster.
    - **Calinski–Harabasz Index** – Mengukur rasio variasi antar cluster.

    Metrik-metrik ini digunakan **sebagai alat bantu pengambilan keputusan**,
    bukan sebagai ukuran kebenaran absolut.
    """)

    card("⚠️ Pembatasan Metodologis dan Interpretasi")
    st.markdown("""
    Aplikasi ini menekankan **interpretasi deskriptif dan eksploratif** terhadap
    hasil clustering. Analisis yang ditampilkan bersifat relatif terhadap data
    yang dianalisis dan **tidak merepresentasikan hubungan kausal maupun prediksi**.

    Penentuan cluster untuk data baru hanya didukung oleh **algoritma tertentu**
    dan diperlakukan sebagai **assignment eksploratif**, bukan supervised prediction.
    """)

    st.info(
        "Catatan: Hasil clustering sangat dipengaruhi oleh pemilihan variabel "
        "dan karakteristik data. Interpretasi hasil harus selalu mempertimbangkan "
        "konteks domain dan keterbatasan metodologis."
    )
