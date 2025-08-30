import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import requests
import branca.colormap as cm
from sklearn.preprocessing import StandardScaler
import hdbscan
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

st.set_page_config(page_title="HDBSCAN Clustering", layout="wide")

st.markdown(
    """
    <style>
    /* Latar belakang putih untuk seluruh aplikasi */
    .stApp {
        background-color: white !important;
        color: black !important;
    }

   /* Sidebar dengan warna gradasi hijau → oranye → merah */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2ecc71, #f39c12, #e74c3c) !important; 
        color: white !important;
    }

    /* Elemen teks di sidebar */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Radio button navigasi */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        display: block;
        background: rgba(255, 255, 255, 0.1);
        padding: 8px 15px;
        margin: 5px 0;
        border-radius: 8px;
        font-weight: 600;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }

    /* Hover effect navigasi */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.25);
        cursor: pointer;
    }

    /* Navigasi aktif (dipilih) */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: white !important;
        color: #27ae60 !important;
        font-weight: 700;
        border: 1px solid #27ae60;
    }

    /* Elemen markdown seperti teks dan paragraf jadi hitam */
    div[data-testid="stMarkdownContainer"] {
        color: black !important;
    }

    /* Header dan teks */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: black !important;
    }

    /* Tabel teks dan isi jadi hitam */
    .stDataFrame div, .stTable div {
        color: black !important;
    }

    /* Tooltip pada map */
    .leaflet-tooltip {
        background-color: white;
        color: black;
        border: 1px solid black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigasi
menu = st.sidebar.radio("📌 Navigasi", ["Home", "Materi", "Klasterisasi"])

# =======================
# HOME
# =======================
if menu == 'Home':
    st.title("👋 Selamat Datang di Website Sederhana Klasterisasi HDBSCAN!")
    st.markdown("""
    Hai! 👋  
    Website ini dirancang untuk membantumu **belajar dan memahami proses klasterisasi data menggunakan algoritma HDBSCAN**, sebuah metode clustering canggih yang mampu mengenali bentuk klaster tidak beraturan dan mengidentifikasi outlier.

    💡 Melalui website ini kamu tidak hanya mencoba, tapi juga belajar dulu konsep dasar metode HDBSCAN!
    
    📘 **Sebelum memulai, baca dulu materinya yaa...**  
    Agar kamu lebih paham bagaimana metode **HDBSCAN bekerja** dalam mengelompokkan data berdasarkan kepadatan.  
    Dengan memahami konsep dasarnya, kamu akan lebih siap untuk **menginterpretasikan hasil klasterisasi** dan mengevaluasi kualitasnya.

    > Yuk mulai perjalanan datamu sekarang!
    """)

    # if st.button("🚀 Mulai Sekarang"):
    #     st.menu = 'Materi'
    #     st.rerun()

# =======================
# MATERI HDBSCAN
# =======================
elif menu == "Materi":
    st.title("📚 Materi: Pemahaman HDBSCAN")

    sub_menu = st.radio("Pilih topik yang ingin kamu pelajari:",
                        ["📖 Pengertian",
                         "⚙️ Proses HDBSCAN",
                         "🔧 Parameter",
                         "✅ Keuntungan & ⚠️ Kelemahan",
                         "📊 Visualisasi"])

    if sub_menu == "📖 Pengertian":
        st.subheader("📖 Apa itu HDBSCAN?")
        st.markdown("""
        HDBSCAN adalah algoritma klasterisasi yang digunakan dalam pembelajaran tanpa pengawasan (*unsupervised learning*) untuk mengidentifikasi kelompok data yang memiliki kemiripan atau yang dikenal sebagai klaster di dalam suatu dataset. Algoritma ini merupakan pengembangan dari algoritma DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*) untuk mengatasi beberapa keterbatasan DBSCAN.
        
        DBSCAN memiliki beberapa kelemahan yang berhasil diatasi oleh HDBSCAN, seperti sensitivitas terhadap parameter epsilon (ε) dan minPts yang harus ditentukan secara tepat agar hasil klasterisasi akurat. Selain itu, DBSCAN tidak efektif saat menghadapi data dengan kepadatan yang bervariasi karena algoritmanya mengasumsikan semua klaster memiliki tingkat kepadatan yang sama. 
        HDBSCAN hadir sebagai solusi dengan menghilangkan kebutuhan penentuan ε secara eksplisit dan menggantinya dengan pendekatan hierarkis yang membentuk struktur klaster berdasarkan kestabilan. HDBSCAN secara otomatis menyesuaikan jumlah klaster berdasarkan pola kepadatan dalam data. HDBSCAN akan mengelompokkan area dengan kepadatan tinggi sebagai klaster, sementara titik-titik yang tersebar atau dengan kepadatan rendah dianggap sebagai *noise* (data luar atau tidak termasuk dalam klaster manapun).

        """)

    elif sub_menu == "⚙️ Proses HDBSCAN":
        st.subheader("⚙️ Proses Kerja HDBSCAN")
        st.markdown("""
        1. Menentukan Parameter Awal
        > Tetapkan beberapa nilai awal untuk parameter ukuran minimum klaster (*min_cluster_size*), misalnya M_pts = 2, 3, 4, 5, dan 6, sebagai dasar eksperimen klasterisasi.
        2. Menghitung Core Distance
        > Untuk setiap titik data, hitung jarak ke tetangga ke-M_pts terdekat. Jika jumlah tetangga tidak mencukupi, titik tersebut dianggap sebagai *noise* dan tidak digunakan sebagai pusat klaster.
        3. Menghitung *Mutual Reachability Distance* (MRD)
        > Tentukan jarak antara semua pasangan titik data menggunakan konsep MRD, yaitu nilai maksimum dari *core distance* titik pertama, *core distance* titik kedua, dan jarak langsung antar titik. Nilai ini memperhitungkan kepadatan lokal masing-masing titik.
        4. Membangun *Minimum Spanning Tree* (MST)
        > Buat graf pohon dengan menghubungkan seluruh titik berdasarkan MRD, dengan tujuan meminimalkan total jarak. MST ini menjadi dasar dalam menyusun hubungan antar titik untuk klasterisasi.
        5. Menyusun Struktur Hierarki Klaster
        > Dari MST, identifikasi klaster dengan memutus tepi-tepi yang memiliki nilai MRD tinggi. Proses ini menghasilkan struktur hierarkis (mirip dendogram) yang menunjukkan potensi pembentukan klaster pada berbagai level.
        6. Memadatkan Pohon Hierarki
        > Meringkas hierarki klaster untuk menyisakan klaster yang stabil. Klaster yang kurang stabil atau tidak konsisten pada kepadatan tertentu akan dieliminasi dan dianggap sebagai *noise*, biasanya diberi label -1.
        7. Evaluasi dengan *Silhouette Coefficient* (SC)
        > Setelah proses klasterisasi selesai untuk tiap nilai M_pts, hitung nilai *Silhouette Coefficient* untuk mengukur seberapa baik data terkelompok. Nilai ini menjadi indikator kualitas klaster yang terbentuk tanpa perlu mengetahui jumlah klaster sebelumnya.

        """)

    elif sub_menu == "🔧 Parameter":
        st.subheader("🔧 Parameter Penting HDBSCAN")
        st.markdown("""
        - **min_cluster_size**: Jumlah minimal data dalam satu klaster.  
          > Semakin besar nilainya, semakin besar pula ukuran minimum klaster yang dibentuk.  
        - **min_samples**: Jumlah minimum tetangga terdekat untuk menganggap titik sebagai bagian dari cluster padat.  
          > Semakin besar nilai ini, semakin ketat definisi kepadatan.
        - **metric**: Metrik jarak yang digunakan untuk mengukur kemiripan antara titik-titik. (`euclidean`, `manhattan`)
        """)

    elif sub_menu == "✅ Keuntungan & ⚠️ Kelemahan":
        st.subheader("✅ Keuntungan HDBSCAN")
        st.markdown("""
        - Tidak butuh jumlah klaster di awal
        - Bisa menemukan klaster dengan bentuk tidak beraturan
        - Deteksi outlier (noise) otomatis
        - Cocok untuk data besar dan tidak terstruktur
        """)

        st.subheader("⚠️ Kelemahan HDBSCAN")
        st.markdown("""
        - Lebih lambat dibanding DBSCAN (karena hierarki) karen cukup berat secara komputasi, terutama untuk dataset berukuran besar.
        - Sulit dioptimasi parameter (min_cluster_size dan min_samples). Nilai parameter yang tidak sesuai dapat memengaruhi kualitas hasil klasterisasi.
        - Hasil klasterisasi sangat dipengaruhi oleh jenis metrik jarak yang digunakan. Jika metrik jarak yang dipilih tidak sesuai dengan struktur alami data, maka hasil klasterisasi bisa menjadi kurang optimal atau bahkan menyesatkan.
        """)

    elif sub_menu == "📊 Visualisasi":
        st.subheader("📊 Visualisasi dalam HDBSCAN")
        st.markdown("""
        HDBSCAN menyediakan beberapa bentuk visualisasi penting:

        - **Condensed Tree**: Menampilkan stabilitas klaster dan cara pemotongan struktur hierarki
        - **Cluster Plot**: Visualisasi distribusi data dalam 2D (jika data bisa direduksi)

        Contoh:
        """)
        st.image("https://blog.renzhamin.com/hdbscan/images/condensed_tree.png", caption="Contoh *Condensed Tree Plot*", use_column_width=False)
        st.markdown("""
        Visualisasi diatas merupakan struktur hierarkis pembentukan klaster berdasarkan kepadatan (*density-based hierarchy*). Sumbu vertikal (λ value) menunjukkan tingkat kepadatan, di mana semakin ke bawah menandakan wilayah dengan kepadatan yang lebih tinggi. Batang horizontal pada plot menggambarkan ukuran klaster, yang ditentukan berdasarkan jumlah anggota di dalamnya—semakin lebar batang, semakin besar klaster tersebut. 
        Warna batang menunjukkan banyaknya anggota, dengan warna lebih cerah (hijau-kuning) menunjukkan jumlah anggota yang lebih banyak.
        Lingkaran dan area yang ditandai dalam plot menunjukkan klaster-klaster yang dipilih secara otomatis oleh HDBSCAN berdasarkan kriteria kestabilan (*cluster stability*).
        """)

    # if st.button("➡️ Coba Klasterisasi"):
    #     st.menu = 'Klasterisasi'
    #     st.rerun()

# =======================
# KLASERISASI
# =======================
elif menu == 'Klasterisasi':
    st.title("🗺️ Klasterisasi HDBSCAN dan Visualisasi di Peta Provinsi Indonesia")
    st.markdown("> Upload data dan mulai proses klasterisasi seperti yang telah kamu pelajari! 🎯")

    # === 1. Load GeoJSON dari URL ===
    @st.cache_data
    def load_geojson():
        url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
        response = requests.get(url)
        prov_geojson = response.json()
        gdf = gpd.GeoDataFrame.from_features(prov_geojson["features"])
        gdf.crs = "epsg:4326"
        gdf['kode'] = gdf['kode'].astype(int)  # penting agar bisa merge
        return gdf

    gdf_map = load_geojson()

    # === 2. Upload Data CSV ===
    uploaded_file = st.file_uploader("📤 Upload Data CSV (harus ada kolom 'kode' dan variabel numerik)", type=["csv"], key="upload_csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file, delimiter=';')
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        #Statistika Deskriptif
        st.subheader("📊 Statistika Deskriptif")
        numeric_all = df.select_dtypes(include=['float64', 'int64']).drop(columns=['kode'])
        st.dataframe(numeric_all.describe())

        st.subheader("📊 Visualisasi Proporsi Kolom-Kolom Numerik")

        # Ambil semua kolom numerik
        numeric_cols_all = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Pilih kolom mana saja yang ingin divisualisasikan sebagai kategori proporsi
        selected_cols = st.multiselect(
            "Pilih Kolom-Kolom Numerik yang Dianggap sebagai Kategori (misalnya Pendidikan atau Sektor Pekerjaan)",
            numeric_cols_all,
            default=['<= SD/MI', 'SMP/MTS', 'SMA/SMK',
                     'Perguruan Tinggi'] if 'Perguruan Tinggi' in df.columns else numeric_cols_all[:3]
        )

        # Tampilkan Pie Chart jika kolom terpilih ≥ 2
        if len(selected_cols) < 2:
            st.warning("⚠️ Pilih minimal 2 kolom numerik untuk membentuk kategori.")
        else:
            # Hitung total nilai tiap kolom (asumsi mewakili jumlah per kategori)
            df_sum = df[selected_cols].sum().reset_index()
            df_sum.columns = ['Kategori', 'Jumlah']

            # Pie chart interaktif
            fig = px.pie(df_sum, names='Kategori', values='Jumlah', title='Distribusi Proporsi per Kategori Terpilih')
            st.plotly_chart(fig)

        # Pastikan ada kolom "kode"
        if 'kode' not in df.columns:
            st.error("❗ Kolom 'kode' (kode provinsi) tidak ditemukan di CSV!")
        else:
            # Pilih kolom numerik untuk clustering
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop(columns=['kode']).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("⚠️ Butuh minimal 2 kolom numerik untuk HDBSCAN.")
            else:
                selected_cols = numeric_cols
                # st.success(
                #     f"✅ Menggunakan {len(selected_cols)} kolom numerik untuk klasterisasi otomatis: {', '.join(selected_cols)}")

                st.subheader("🔍 Hasil Standarisasi")

                # Standarisasi + HDBSCAN
                X_scaled = StandardScaler().fit_transform(df[selected_cols])
                # Konversi ke DataFrame biar bisa tampil tabelnya
                scaled_df = pd.DataFrame(X_scaled, columns=selected_cols)
                st.dataframe(scaled_df)

                lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
                y_pred = lof.fit_predict(X_scaled)
                lof_scores = -lof.negative_outlier_factor_

                # Tandai outlier
                df['Outlier_LOF'] = y_pred  # -1 = outlier, 1 = inlier

                # Visualisasi di Streamlit
                st.subheader("🔍 Deteksi Outlier dengan LOF")
                lof_df = df[['Provinsi', 'Outlier_LOF']].copy()
                lof_df['Skor LOF'] = lof_scores
                lof_df['Status'] = lof_df['Outlier_LOF'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')
                st.dataframe(lof_df)

                st.subheader("🗺️ Proses Klasterisasi")

                min_cluster_size = st.slider("Pilih Min_Cluster_Size", 2, 6, 3)
                min_samples = st.slider("Pilih Min_Samples", 1, 10, 2)
                metric_option = st.selectbox(
                    "📐 Pilih Metode Pengukuran Jarak (Metric)",
                    options=["euclidean", "manhattan"],
                    index=0
                )

                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric_option)
                labels = clusterer.fit_predict(X_scaled)
                df['Cluster'] = labels

                # Evaluasi Hasil Klasterisasi (tanpa noise / cluster -1)
                mask = df['Cluster'] != -1
                if len(set(df['Cluster'][mask])) > 1:
                    score = silhouette_score(X_scaled[mask], df['Cluster'][mask])
                    st.info(f"📈 **Silhouette Score (tanpa noise)**: `{score:.3f}`")
                else:
                    st.warning("⚠️ Silhouette Score tidak dapat dihitung (hanya 1 cluster atau hanya noise).")

                # Gabungkan ke GeoDataFrame
                merged = gdf_map.merge(df[['kode', 'Cluster']], on='kode', how='left')

                # Warna berdasarkan cluster
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                st.success(f"✅ Jumlah Klaster Terbentuk: {n_clusters}")

                # Pilihan jenis visualisasi
                st.subheader("📊 Pilih Jenis Visualisasi")
                plot_type = st.radio("Pilih jenis plot:", ["Tree Plot", "Scatter Plot"])
                
                if plot_type == "Tree Plot":
                    # Buat plot pohon dari hasil clustering
                    st.subheader("🌳 Visualisasi Condensed Tree dari HDBSCAN")
                    plt.figure(figsize=(6, 4))
                    clusterer.condensed_tree_.plot(select_clusters=True, 
                                                   selection_palette=sns.color_palette("deep"))
                    fig = plt.gcf()
                    st.pyplot(fig, use_container_width=False)
                
                elif plot_type == "Scatter Plot":
                    # PCA untuk reduksi dimensi
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(X_scaled)
                    
                    plt.figure(figsize=(12, 8))
                    unique_labels = set(labels)
                    palette = sns.color_palette("Set2", len(unique_labels) - (1 if -1 in unique_labels else 0))
                    
                    for label in unique_labels:
                        idx = labels == label
                        points = reduced[idx]
                    
                        if label == -1:
                            # Noise
                            plt.scatter(points[:, 0], points[:, 1], s=100, c='gray', 
                                        label='Noise', edgecolor='k', alpha=0.6)
                        else:
                            color = palette[label]
                            plt.scatter(points[:, 0], points[:, 1], s=60, color=color, 
                                        edgecolor='k', label=f'Klaster {label}', alpha=0.9)
                    
                            if len(points) >= 3:
                                hull = ConvexHull(points)
                                hull_points = points[hull.vertices]
                                polygon = Polygon(hull_points, closed=True, facecolor=color, 
                                                  alpha=0.2, edgecolor=color, linewidth=2)
                                plt.gca().add_patch(polygon)
                    
                    # Tambahan estetika seperti peta
                    plt.gca().set_facecolor('#f7f7f7')
                    plt.title("Distribusi Klaster HDBSCAN", fontsize=14, pad=20)  # pakai pad untuk jarak
                    plt.xlabel("Komponen 1", fontsize=12)
                    plt.ylabel("Komponen 2", fontsize=12)
                    plt.xticks([])
                    plt.yticks([])
                    plt.box(True)
                    plt.legend(title='Klaster', loc='best')
                    
                    # Atur jarak atas
                    plt.subplots_adjust(top=0.9)  # 0.9 bisa diganti 0.85/0.95 sesuai kebutuhan
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                from branca.colormap import LinearColormap

                min_val = merged['Cluster'].min()
                max_val = merged['Cluster'].max()

                colors = ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45']  # gradasi hijau terang ke gelap
                colormap = LinearColormap(colors, vmin=min_val, vmax=max_val)

                # colormap = cm.linear.YlGn_r.scale(merged['Cluster'].min(), merged['Cluster'].max())
                colormap.caption = 'Klaster HDBSCAN'

                # Buat Folium Map
                m = folium.Map(location=[-2, 118], zoom_start=5, tiles='cartodbpositron')

                folium.GeoJson(
                    merged,
                    style_function=lambda feature: {
                        'fillColor': colormap(feature['properties']['Cluster']) if feature['properties']['Cluster'] is not None else 'gray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['Propinsi', 'Cluster'],
                        aliases=['Provinsi:', 'Klaster:'],
                        localize=True
                    )
                ).add_to(m)

                colormap.add_to(m)
                st.subheader("🗺️ Peta Interaktif Klasterisasi")
                st_data = st_folium(m, width=1000)

                st.subheader("📋 Tabel Hasil Klasterisasi")

                # Gabungkan nama provinsi
                if 'Provinsi' in df.columns:
                    cluster_table = df[['kode', 'Provinsi', 'Cluster']].sort_values(by='Cluster')
                    st.dataframe(cluster_table, use_container_width=True)
                else:
                    st.warning("⚠️ Kolom 'Provinsi' tidak ditemukan di data.")

                # Kolom yang akan diringkas
                summary_cols = [
                    'Usia Produktif', '<= SD/MI', 'SMP/MTS', 'SMA/SMK',
                    'Perguruan Tinggi', 'Primer', 'Sekunder', 'Tersier'
                ]

                # Pastikan hanya ambil kolom yang ada
                available_cols = [col for col in summary_cols if col in df.columns]

                # 1. Buat ringkasan per klaster (pakai sum karena data absolut)
                summary_df = df.groupby('Cluster')[available_cols].sum()

                # 2. Hitung jumlah provinsi per klaster
                summary_df['Jumlah Provinsi'] = df.groupby('Cluster').size()

                # 3. Hitung total untuk pendidikan dan sektor per klaster
                pendidikan_cols = ['<= SD/MI', 'SMP/MTS', 'SMA/SMK', 'Perguruan Tinggi']
                sektor_cols = ['Primer', 'Sekunder', 'Tersier']

                summary_df['Total Pendidikan'] = summary_df[pendidikan_cols].sum(axis=1)
                summary_df['Total Sektor'] = summary_df[sektor_cols].sum(axis=1)

                # 4. Hitung persentase per kategori di dalam masing-masing klaster
                for col in pendidikan_cols:
                    summary_df[col + " (%)"] = (summary_df[col] / summary_df['Total Pendidikan']) * 100

                for col in sektor_cols:
                    summary_df[col + " (%)"] = (summary_df[col] / summary_df['Total Sektor']) * 100

                # 5. Hitung persentase Usia Produktif terhadap total keseluruhan
                total_usia = summary_df['Usia Produktif'].sum()
                summary_df['Usia Produktif (%)'] = (summary_df['Usia Produktif'] / total_usia) * 100

                # 6. Hapus kolom helper + absolut yang tidak ingin ditampilkan
                summary_df = summary_df.drop(
                    columns=pendidikan_cols + sektor_cols + ['Total Pendidikan', 'Total Sektor'])

                # Reset index dan atur urutan kolom
                summary_df = summary_df.reset_index()
                summary_df = summary_df[['Cluster', 'Jumlah Provinsi', 'Usia Produktif (%)',
                                         '<= SD/MI (%)', 'SMP/MTS (%)', 'SMA/SMK (%)', 'Perguruan Tinggi (%)',
                                         'Primer (%)', 'Sekunder (%)', 'Tersier (%)']]

                # print(summary_df)

                # Tampilkan
                st.subheader("📊 Ringkasan Statistik Persentase per Klaster")
                st.dataframe(summary_df, use_container_width=True)

                st.subheader("🧠 Interpretasi Otomatis Tiap Klaster")

                for _, row in summary_df.iterrows():
                    cluster_id = int(row['Cluster'])
                    jumlah_prov = int(row['Jumlah Provinsi'])

                    if cluster_id == -1:
                        st.markdown(f"""
                **Klaster {cluster_id} (Noise)**  
                Klaster ini terdiri dari {jumlah_prov} provinsi yang **tidak dimasukkan ke dalam klaster manapun oleh HDBSCAN**.  
                Provinsi-provinsi ini memiliki karakteristik yang dianggap **berbeda secara signifikan** dari mayoritas lainnya.  
                Hal ini bisa disebabkan oleh **kombinasi unik dalam pendidikan, struktur ekonomi, atau distribusi usia produktif** yang tidak menyerupai klaster manapun.  
                Wilayah ini bisa mencerminkan provinsi-provinsi **metropolitan, ekstrem, atau sangat heterogen**, dan layak diteliti lebih lanjut secara individual.
                """)
                        continue

                    # Agregasi pendidikan
                    pend_rendah = row['<= SD/MI (%)']
                    pend_menengah = row['SMP/MTS (%)'] + row['SMA/SMK (%)']
                    pend_tinggi = row['Perguruan Tinggi (%)']

                    pendidikan_agregat = {
                        'rendah': pend_rendah,
                        'menengah': pend_menengah,
                        'tinggi': pend_tinggi
                    }

                    kategori_pendidikan = max(pendidikan_agregat, key=pendidikan_agregat.get)

                    # Usia produktif kategori
                    usia = row['Usia Produktif (%)']
                    if usia >= 40:
                        usia_label = "tinggi"
                    elif usia >= 20:
                        usia_label = "sedang"
                    else:
                        usia_label = "rendah"

                    # Sektor dominan
                    sektor_cols = ['Primer (%)', 'Sekunder (%)', 'Tersier (%)']
                    sektor_dominan = max(sektor_cols, key=lambda x: row[x])
                    sektor_dominan = sektor_dominan.replace(" (%)", "")

                    # Susun interpretasi
                    interpretasi = f"""
                **Klaster {cluster_id}**  
                Klaster ini terdiri dari {jumlah_prov} provinsi, dengan karakteristik penduduk usia produktif yang tergolong **{usia_label}**.  
                Dilihat dari tingkat pendidikan, klaster ini didominasi oleh pendidikan **{kategori_pendidikan}**.  
                Sektor ekonomi yang paling dominan di klaster ini adalah sektor **{sektor_dominan.lower()}**.
                """

                    # Tambahan penilaian opsional
                    if kategori_pendidikan == 'rendah' and usia_label == 'tinggi':
                        interpretasi += "\n➡️ Wilayah ini memiliki tenaga kerja usia produktif yang besar, namun masih didominasi oleh pendidikan rendah. Ini bisa menjadi target prioritas untuk peningkatan kualitas SDM."
                    elif kategori_pendidikan == 'menengah' and usia_label == 'tinggi':
                        interpretasi += "\n➡️ Wilayah ini memiliki tenaga kerja usia produktif yang besar dan struktur pendidikan yang cukup baik, namun peningkatan ke jenjang pendidikan tinggi masih diperlukan untuk mendukung pengembangan SDM jangka panjang"
                    elif kategori_pendidikan == 'menengah' and usia_label == 'rendah':
                        interpretasi += "\n➡️ Meski struktur pendidikan sudah cukup baik, proporsi usia produktif masih rendah. Hal ini bisa menunjukkan tantangan regenerasi tenaga kerja."
                    elif kategori_pendidikan == 'tinggi' and usia_label == 'tinggi':
                        interpretasi += "\n➡️ Wilayah ini menunjukkan kesiapan SDM yang kuat, baik dari segi pendidikan maupun usia produktif — potensi besar dalam pembangunan ekonomi modern."
                    elif usia_label == 'rendah':
                        interpretasi += "\n➡️ Proporsi usia produktif yang rendah dapat menjadi perhatian, terutama jika ingin mendorong produktivitas wilayah secara jangka panjang."
                    else:
                        interpretasi += "\n➡️ Klaster ini menunjukkan komposisi yang relatif seimbang antara pendidikan dan usia produktif."

                    st.markdown(interpretasi)

                # Download hasil klasterisasi
                merged_df = df[['kode', 'Cluster']]
                csv = merged_df.to_csv(index=False)

                st.download_button("⬇️ Download Hasil Klasterisasi", csv, "hasil_klaster.csv", "text/csv")










