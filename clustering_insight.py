import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


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
# MAIN FUNCTION
# ==================================================
def cluster_insight(df, features, labels):

    # ==================================================
    # VALIDASI INPUT
    # ==================================================
    if df is None or labels is None:
        st.info("Silakan jalankan clustering terlebih dahulu pada tahap sebelumnya.")
        return

    df_cluster = df.copy()
    df_cluster["Cluster"] = labels

    # Abaikan noise (DBSCAN / HDBSCAN)
    df_valid = df_cluster[df_cluster["Cluster"] != -1]

    if df_valid["Cluster"].nunique() < 2:
        st.warning(
            "Jumlah cluster valid kurang dari dua. "
            "Insight tidak dapat ditampilkan secara bermakna."
        )
        return

    # ==================================================
    # 1. KONTEKS INTERPRETASI
    # ==================================================
    card(
        "🎯 Konteks Interpretasi",
        "Pengaturan ini digunakan untuk menyesuaikan interpretasi hasil clustering "
        "berdasarkan indikator utama dan konteks domain (opsional)."
    )

    use_manual = st.checkbox(
        "Gunakan konteks domain manual",
        value=False
    )

    if use_manual:
        context_features = st.multiselect(
            "Pilih indikator utama",
            features,
            default=features
        )

        direction = st.radio(
            "Arah interpretasi nilai tinggi",
            ["Netral (relatif)", "Tinggi = Positif", "Tinggi = Negatif"]
        )
    else:
        context_features = features
        direction = "Netral (relatif)"

    # ==================================================
    # 2. RINGKASAN STRUKTUR CLUSTER
    # ==================================================
    card(
        "📌 Ringkasan Struktur Cluster",
        "Menunjukkan jumlah dan proporsi observasi pada setiap cluster "
        "yang terbentuk setelah proses clustering."
    )

    summary = (
        df_valid.groupby("Cluster")
        .size()
        .reset_index(name="Jumlah Observasi")
    )
    summary["Proporsi (%)"] = (
        summary["Jumlah Observasi"] / summary["Jumlah Observasi"].sum() * 100
    ).round(2)

    st.dataframe(summary, use_container_width=True)

    # ==================================================
    # 3. PROFIL STATISTIK CLUSTER
    # ==================================================
    card(
        "📊 Profil Statistik per Cluster",
        "Ringkasan nilai rata-rata dan deviasi standar setiap variabel "
        "untuk memahami perbedaan karakteristik antar cluster."
    )

    profile_mean = df_valid.groupby("Cluster")[features].mean()
    profile_std = df_valid.groupby("Cluster")[features].std()

    profile = (
        pd.concat({"Mean": profile_mean, "Std": profile_std}, axis=1)
        .round(3)
        .reset_index()
    )

    st.dataframe(profile, use_container_width=True)

    # ==================================================
    # 4. PENENTUAN LABEL RELATIF CLUSTER
    # ==================================================
    z_df = df_valid.copy()
    z_df[context_features] = z_df[context_features].apply(zscore)

    cluster_score = (
        z_df.groupby("Cluster")[context_features]
        .mean()
        .mean(axis=1)
    )

    ranked = cluster_score.sort_values().index.tolist()
    n = len(ranked)

    if n == 2:
        base_labels = ["Rendah", "Tinggi"]
    elif n == 3:
        base_labels = ["Rendah", "Sedang", "Tinggi"]
    else:
        base_labels = [f"Level {i+1}" for i in range(n)]

    label_map = {cid: base_labels[i] for i, cid in enumerate(ranked)}

    if direction == "Tinggi = Negatif":
        label_map = dict(zip(label_map.keys(), reversed(label_map.values())))

    df_valid["Cluster_Label"] = df_valid["Cluster"].map(label_map)

    # ==================================================
    # 5. KUSTOMISASI NAMA CLUSTER
    # ==================================================
    card(
        "✏️ Kustomisasi Nama Cluster",
        "Penamaan cluster dilakukan oleh pengguna berdasarkan konteks domain "
        "dan interpretasi analitik."
    )

    custom_names = {}
    for cid in ranked:
        default_name = f"Cluster {cid} ({label_map[cid]})"
        custom_names[cid] = st.text_input(
            f"Nama untuk Cluster {cid}",
            value=default_name
        )

    df_valid["Cluster_Name"] = df_valid["Cluster"].map(custom_names)

    # ==================================================
    # 6. INTERPRETASI KARAKTERISTIK CLUSTER
    # ==================================================
    card(
        "🧠 Interpretasi Karakteristik Cluster",
        "Interpretasi relatif terhadap rata-rata global menggunakan pendekatan "
        "z-score untuk mengidentifikasi karakteristik dominan setiap cluster."
    )

    global_mean = df_valid[features].mean()
    global_std = df_valid[features].std()
    Z_THRESHOLD = 0.5

    for cid in ranked:
        traits = []
        for f in features:
            z = (profile_mean.loc[cid, f] - global_mean[f]) / global_std[f]
            if z > Z_THRESHOLD:
                traits.append(f"{f} relatif tinggi")
            elif z < -Z_THRESHOLD:
                traits.append(f"{f} relatif rendah")

        trait_text = "; ".join(traits) if traits else "Tidak terdapat karakteristik ekstrem."

        size = summary.loc[
            summary["Cluster"] == cid, "Jumlah Observasi"
        ].values[0]

        st.markdown(
            f"""
            **{custom_names[cid]}**  
            Ukuran cluster: {size} observasi  
            Karakteristik utama: {trait_text}
            """
        )

    # ==================================================
    # 7. VISUALISASI PASCA CLUSTERING
    # ==================================================
    card(
        "📈 Visualisasi Pasca Clustering",
        "Scatter plot dua dimensi untuk mengeksplorasi pemisahan cluster "
        "berdasarkan pasangan variabel numerik terpilih."
    )

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Sumbu X", features)
    with col2:
        y_var = st.selectbox("Sumbu Y", features, index=1)

    fig, ax = plt.subplots()
    for name in df_valid["Cluster_Name"].unique():
        subset = df_valid[df_valid["Cluster_Name"] == name]
        ax.scatter(
            subset[x_var],
            subset[y_var],
            label=name,
            alpha=0.7
        )

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.legend()
    st.pyplot(fig)

    # ==================================================
    # 8. CLUSTER ASSIGNMENT (DATA BARU)
    # ==================================================
    card(
        "🔎 Penentuan Cluster untuk Data Baru",
        "Assignment cluster untuk observasi baru menggunakan model aktif. "
        "Perlu diperhatikan bahwa ini bukan supervised prediction."
    )

    if (
        "model" not in st.session_state
        or "scaler" not in st.session_state
        or "method_name" not in st.session_state
    ):
        st.info("Model clustering aktif belum tersedia.")
        return

    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    method = st.session_state["method_name"]

    SUPPORTED = ["KMeans", "MiniBatchKMeans", "GaussianMixture", "Birch"]
    APPROXIMATE = ["DBSCAN", "HDBSCAN"]
    UNSUPPORTED = ["AgglomerativeClustering", "SpectralClustering", "OPTICS"]

    if method in UNSUPPORTED:
        st.warning(
            f"Metode {method} tidak mendukung penentuan cluster untuk data baru."
        )
        return

    input_data = {}
    for f in features:
        input_data[f] = st.number_input(
            f,
            value=float(df[f].mean())
        )

    if st.button("Tentukan Cluster"):
        x_new = pd.DataFrame([input_data])
        x_scaled = scaler.transform(x_new)

        if method in SUPPORTED:
            cluster_id = model.predict(x_scaled)[0]
            st.success(
                f"Data baru paling sesuai dengan: **{custom_names[cluster_id]}**"
            )

        elif method in APPROXIMATE:
            centers = (
                df_valid.groupby("Cluster")[features]
                .mean()
                .values
            )

            dist = pairwise_distances(x_new[features], centers)
            nearest = dist.argmin()

            st.warning(
                f"Aproksimasi: data paling dekat dengan "
                f"**{custom_names[nearest]}**. "
                "Untuk metode density-based, data dapat dianggap noise."
            )

    # ==================================================
    # 9. CATATAN METODOLOGIS
    # ==================================================
    card(
        "⚠️ Catatan Metodologis",
        "Clustering bersifat unsupervised dan tidak menghasilkan inferensi kausal. "
        "Hasil bersifat eksploratif dan sangat bergantung pada pemilihan variabel "
        "serta karakteristik data."
    )
