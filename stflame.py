# HOME = PETA INTERAKTIF
# HASIL CLUSTERING = HASIL CLUSTERING DARI DATA

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import folium
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from folium.plugins import HeatMap, Fullscreen, MiniMap, MeasureControl, MousePosition
from streamlit_folium import st_folium

def format_tanggal_indo(tgl):
    bulan = {
        1: "Januari",
        2: "Februari",
        3: "Maret",
        4: "April",
        5: "Mei",
        6: "Juni",
        7: "Juli",
        8: "Agustus",
        9: "September",
        10: "Oktober",
        11: "November",
        12: "Desember"
    }

    return f"{tgl.day} {bulan[tgl.month]} {tgl.year}"

tgl = pd.to_datetime("2026-04-28")

# CONFIG
st.set_page_config(
    page_title="Dashboard Hotspot Kebakaran Hutan di Provinsi Jawa Timur",
    layout="wide"
)

# SIDEBAR MENU
st.sidebar.title("📂 Dashboard")

menu = st.sidebar.radio(
    "Pilih Halaman",
    ["🏠 Dashboard", "📈 Visualisasi Data", "📊 Hasil Clustering", "🔮 Forecasting & Anomali"]
)

# LOAD DATA
@st.cache_data
def load_cluster():
    return pd.read_csv("hasil_cluster_hotspot.csv")

@st.cache_resource
def load_model():
    model = joblib.load("model_kmeans_hotspot.pkl")
    scaler = joblib.load("scaler_hotspot.pkl")
    return model, scaler

@st.cache_data
def load_aktual():
    files = [
        "data karhut aktual tahun 2021.csv",
        "data karhut aktual tahun 2022.csv",
        "data karhut aktual tahun 2023.csv"
    ]
    df_list = []
    for f in files:
        temp = pd.read_csv(f, sep=",")
        temp.columns = temp.columns.str.strip()
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)
    df.columns = ["Kabupaten", "Tanggal", "Luas_Ha"]

    bulan_indonesia = {
        "Januari": "January", "Februari": "February", "Maret": "March",
        "April": "April", "Mei": "May", "Juni": "June",
        "Juli": "July", "Agustus": "August", "September": "September",
        "Oktober": "October", "November": "November", "Desember": "December"
    }

    def parse_date(d):
        d = str(d)
        for indo, eng in bulan_indonesia.items():
            if indo in d:
                d = d.replace(indo, eng)
                break
        return pd.to_datetime(d, format="%d %B %Y", errors="coerce")

    df["Tanggal"] = df["Tanggal"].astype(str).apply(parse_date)

    df["Kabupaten"] = df["Kabupaten"].astype(str).str.strip().str.upper()

    mapping_kab = {
        "SARADAN": "MADIUN",
        "BANYUWANGI UTARA": "BANYUWANGI",
        "JATIROGO": "TUBAN",
        "LAWU DS": "NGAWI",
        "PADANGAN": "BOJONEGORO",
        "PARENGAN": "TUBAN",
        "MADURA": "BANGKALAN"
    }
    df["Kabupaten"] = df["Kabupaten"].replace(mapping_kab)

    return df

@st.cache_data
def load_map():
    df = pd.read_csv(
        "Data Kebakaran 5 Tahun Terakhir.csv",
        sep=","
    )

    df["Tanggal"] = pd.to_datetime(
    df["Tanggal"],
    format="%Y-%m-%d",   # jika data awal format tahun-bulan-tanggal
    errors="coerce"
    )

    df.columns = df.columns.str.strip()

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    df["Tanggal"] = pd.to_datetime(
        df["Tanggal"],
        errors="coerce",
        dayfirst=True
    )

    df = df.dropna(subset=["Latitude", "Longitude", "Tanggal"])

    return df

# HOME
if menu == "🏠 Dashboard":

    st.title("🔥 Hotspot yang Terindikasi Menjadi Kebakaran Hutan yang Terdeteksi Satelit di Provinsi Jawa Timur")

    df_map = load_map()

    # filter tanggal
    col1, col2 = st.columns(2)

    with col1:
        tgl_awal = st.date_input(
            "📅 Tanggal Awal",
            value=df_map["Tanggal"].min().date(),
            min_value=pd.to_datetime("2021-01-01").date(),
            max_value=pd.to_datetime("2025-12-31").date()
        )

    with col2:
        tgl_akhir = st.date_input(
            "📅 Tanggal Akhir",
            value=df_map["Tanggal"].max().date(),
            min_value=pd.to_datetime("2021-01-01").date(),
            max_value=pd.to_datetime("2025-12-31").date()
        )

    df_map = df_map[
        (df_map["Tanggal"].dt.date >= tgl_awal) &
        (df_map["Tanggal"].dt.date <= tgl_akhir)
    ]
    df_map["Waktu"] = df_map["Waktu"].astype(str).str.strip()

    # kalau ada tulisan WIB hapus
    df_map["Waktu"] = df_map["Waktu"].str.replace(" WIB", "", regex=False)

    st.write("Jumlah hotspot:", len(df_map))

    # warna
    def warna(conf):
        conf = str(conf)

        if conf == "3":
            return "red"
        elif conf == "2":
            return "orange"
        else:
            return "yellow"

    # peta
    m = folium.Map(
        location=[-7.7, 112.8],
        zoom_start=8,
        tiles="OpenStreetMap"
    )

    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MeasureControl().add_to(m)
    MousePosition().add_to(m)

    legend_html = """
    <div style="
    position: fixed;
    bottom: 40px;
    left: 40px;
    z-index: 9999;
    background: rgba(255,255,255,0.95);
    padding: 14px 16px;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    font-size: 14px;
    line-height: 1.6;
    color: black;
    min-width: 340px;
    border: 1px solid #ddd;
    ">

    <div style="font-weight:bold; font-size:15px; margin-bottom:8px; color:black;">
    🔥 Keterangan Confidence
    </div>

    <div style="color:black;">
    <span style="color:red; font-size:18px;">●</span>
    Merah = Confidence 3 (High / Titik Api dengan Tingkat Tinggi)
    </div>

    <div style="color:black;">
    <span style="color:orange; font-size:18px;">●</span>
    Orange = Confidence 2 (Medium / Titik Api dengan Tingkat Sedang)
    </div>

    <div style="color:black;">
    <span style="color:gold; font-size:18px;">●</span>
    Kuning = Confidence 1 (Low / Titik Api dengan Tingkat Rendah)
    </div>

    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # marker
    for _, row in df_map.iterrows():

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color=warna(row["Confidence"]),
            fill=True,
            fill_color=warna(row["Confidence"]),
            fill_opacity=0.75,
            popup=f"""
            <b>Kabupaten:</b> {row['Kabupaten']}<br>
            <b>Tanggal:</b> {format_tanggal_indo(row['Tanggal'])}<br>
            <b>Jam:</b> {row['Waktu']}<br>
            <b>Tingkatan:</b> {row['Confidence']}
            """,
            tooltip=row["Kabupaten"]
        ).add_to(m)

    HeatMap(
        df_map[["Latitude","Longitude"]].values.tolist(),
        radius=18,
        blur=15
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(
        m,
        width=1300,
        height=750
    )

    st.subheader("📋 Keterangan")

    # RINGKASAN
    total_hotspot = len(df_map)
    jumlah_kabupaten = df_map["Kabupaten"].nunique()

    high = (df_map["Confidence"].astype(str) == "3").sum()
    medium = (df_map["Confidence"].astype(str) == "2").sum()
    low = (df_map["Confidence"].astype(str) == "1").sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Hotspot", total_hotspot)
    c2.metric("Kabupaten Terdampak", jumlah_kabupaten)
    c3.metric("Confidence Tinggi", high)
    c4.metric("Confidence Sedang", medium)
    c5.metric("Confidence Rendah", low)

    # KABUPATEN TERBANYAK
    st.markdown("### 🔥 Kabupaten dengan Titik Terbanyak")

    # pastikan confidence string
    df_map["Confidence"] = df_map["Confidence"].astype(str)

    # hitung total per kabupaten
    total = df_map.groupby("Kabupaten").size()

    # hitung per confidence
    low = df_map[df_map["Confidence"] == "1"].groupby("Kabupaten").size()
    medium = df_map[df_map["Confidence"] == "2"].groupby("Kabupaten").size()
    high = df_map[df_map["Confidence"] == "3"].groupby("Kabupaten").size()

    # gabungkan
    top_kab = pd.DataFrame({
        "Jumlah Hotspot": total,
        "Low": low,
        "Medium": medium,
        "High": high
    }).fillna(0)

    # reset index
    top_kab = top_kab.reset_index()

    # ubah ke int biar ga 0.0
    top_kab[["Low", "Medium", "High"]] = top_kab[["Low", "Medium", "High"]].astype(int)

    # urutkan
    top_kab = top_kab.sort_values(by="Jumlah Hotspot", ascending=False)

    # nomor mulai dari 1
    top_kab.index = range(1, len(top_kab) + 1)

    st.dataframe(top_kab.head(10), use_container_width=True)

    # KESIMPULAN
    terparah = top_kab.iloc[0]["Kabupaten"]
    jumlah_terparah = top_kab.iloc[0]["Jumlah Hotspot"]

    low_terparah = top_kab.iloc[0]["Low"]
    medium_terparah = top_kab.iloc[0]["Medium"]
    high_terparah = top_kab.iloc[0]["High"]

    st.markdown("### 📝 Kesimpulan")

    st.info(
        f"""
    Pada rentang tanggal **{format_tanggal_indo(pd.to_datetime(tgl_awal))}** hingga **{format_tanggal_indo(pd.to_datetime(tgl_akhir))}**, terdeteksi sebanyak
    **{total_hotspot} titik api** yang tersebar di **{jumlah_kabupaten} kabupaten**.
    Wilayah dengan titik yang sering terdeteksi titik apinya adalah **{terparah}**
    sebanyak **{jumlah_terparah} titik api**.
    """
    )

    # DATA KEBAKARAN AKTUAL
    st.markdown("---")
    st.header("🔥 Data Kebakaran Aktual (Karhut) 2021-2023")

    df_aktual = load_aktual()

    # Aggregate by Kabupaten
    df_agg = df_aktual.groupby("Kabupaten").agg(
        Total_Luas_Ha=("Luas_Ha", "sum"),
        Jumlah_Kejadian=("Luas_Ha", "count"),
        Rata_Luas_Ha=("Luas_Ha", "mean")
    ).reset_index().sort_values("Total_Luas_Ha", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Kabupaten Terbakar", len(df_agg))
    c2.metric("Total Luas Terbakar (Ha)", f"{df_agg['Total_Luas_Ha'].sum():.1f}")
    c3.metric("Total Kejadian", int(df_agg["Jumlah_Kejadian"].sum()))

    # Barplot
    st.subheader("📊 Luas Kebakaran Aktual per Kabupaten")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df_agg.head(15),
        x="Kabupaten",
        y="Total_Luas_Ha",
        color="tomato",
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Total Luas (Ha)")
    st.pyplot(fig)

    # Tabel
    with st.expander("📋 Detail Data Kebakaran Aktual", expanded=False):
        st.dataframe(
            df_agg.reset_index(drop=True),
            use_container_width=True
        )

    kab_terbesar = df_agg.iloc[0]["Kabupaten"]
    luas_terbesar = df_agg.iloc[0]["Total_Luas_Ha"]
    st.info(f"Kabupaten dengan luas kebakaran terbesar adalah **{kab_terbesar}** dengan **{luas_terbesar:.1f} Ha**.")

    # PERBANDINGAN HOTSPOT vs AKTUAL PER KABUPATEN
    st.markdown("---")
    st.header("📊 Perbandingan Hotspot vs Kebakaran Aktual per Kabupaten")

    # Aggregate hotspot per kabupaten (dari data yang sudah difilter tanggal)
    hotspot_kab = df_map.groupby("Kabupaten").size().reset_index(name="Hotspot")

    # Merge hotspot with actual fire aggregation
    perbandingan = hotspot_kab.merge(df_agg, on="Kabupaten", how="outer").fillna(0)
    perbandingan = perbandingan.sort_values("Hotspot", ascending=False).reset_index(drop=True)

    # Hanya tampilkan kab yang punya data hotspot atau aktual
    perbandingan = perbandingan[(perbandingan["Hotspot"] > 0) | (perbandingan["Total_Luas_Ha"] > 0)]

    c1, c2 = st.columns(2)
    c1.metric("Kabupaten dengan Hotspot", len(perbandingan[perbandingan["Hotspot"] > 0]))
    c2.metric("Kabupaten dengan Aktual", len(perbandingan[perbandingan["Total_Luas_Ha"] > 0]))

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(perbandingan))
    width = 0.35

    ax.bar([i - width/2 for i in x], perbandingan["Hotspot"], width, label="Hotspot", color="orange", alpha=0.8)
    ax.bar([i + width/2 for i in x], perbandingan["Jumlah_Kejadian"], width, label="Kejadian Aktual", color="red", alpha=0.8)

    ax.set_xlabel("Kabupaten")
    ax.set_ylabel("Jumlah")
    ax.set_xticks(x)
    ax.set_xticklabels(perbandingan["Kabupaten"], rotation=90, fontsize=8)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("📋 Data Perbandingan per Kabupaten", expanded=False):
        st.dataframe(
            perbandingan.rename(columns={
                "Jumlah_Kejadian": "Kejadian Aktual",
                "Total_Luas_Ha": "Luas Terbakar (Ha)",
                "Rata_Luas_Ha": "Rata Luas (Ha)"
            }),
            use_container_width=True
        )

    # LAPORAN HARIAN HOTSPOT vs AKTUAL
    st.markdown("---")
    st.header("📋 Laporan Harian: Hotspot vs Kebakaran Aktual")

    st.markdown("Laporan ini mencocokkan titik hotspot dengan kejadian kebakaran aktual per tanggal dan kabupaten. Walaupun tidak selalu tepat, data ini bisa jadi perkiraan keterkaitan antara deteksi satelit dan kejadian di lapangan.")

    # Aggregate hotspot per hari per kabupaten
    df_harian_hs = df_map.copy()
    df_harian_hs["Tanggal"] = df_harian_hs["Tanggal"].dt.date
    harian_hs = df_harian_hs.groupby(["Tanggal", "Kabupaten"]).size().reset_index(name="Jumlah_Hotspot")

    # Aggregate aktual per hari per kabupaten
    df_harian_ak = df_aktual.dropna(subset=["Tanggal"]).copy()
    df_harian_ak["Tanggal"] = df_harian_ak["Tanggal"].dt.date
    harian_ak = df_harian_ak.groupby(["Tanggal", "Kabupaten"]).agg(
        Kejadian_Aktual=("Luas_Ha", "count"),
        Luas_Terbakar=("Luas_Ha", "sum")
    ).reset_index()

    # Merge: hotspot sebagai basis
    laporan = harian_hs.merge(harian_ak, on=["Tanggal", "Kabupaten"], how="left").fillna(0)

    # Kolom status
    def status(row):
        if row["Kejadian_Aktual"] > 0:
            return "✅ Cocok"
        return "⚠️ Hotspot saja"

    laporan["Status"] = laporan.apply(status, axis=1)
    laporan = laporan.sort_values(["Tanggal", "Kabupaten"], ascending=[False, True]).reset_index(drop=True)

    # Filter by date range
    laporan = laporan[
        (laporan["Tanggal"] >= tgl_awal) &
        (laporan["Tanggal"] <= tgl_akhir)
    ]

    # Summary metrics
    total_hari = laporan["Tanggal"].nunique()
    total_cocok = len(laporan[laporan["Status"] == "✅ Cocok"])
    total_hs_saja = len(laporan[laporan["Status"] == "⚠️ Hotspot saja"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Hari dengan Hotspot", total_hari)
    col2.metric("Hari Hotspot + Aktual", total_cocok)
    col3.metric("Hari Hotspot Saja", total_hs_saja)

    # Filter by kabupaten
    kab_list = sorted(laporan["Kabupaten"].unique())
    pilih_kab = st.multiselect("Filter Kabupaten", kab_list, default=[])

    if pilih_kab:
        laporan = laporan[laporan["Kabupaten"].isin(pilih_kab)]

    # Tampilkan
    def warnai_status(val):
        if val == "✅ Cocok":
            return "background-color: #d4edda"
        elif val == "⚠️ Hotspot saja":
            return "background-color: #fff3cd"
        return ""

    st.dataframe(
        laporan.style.map(warnai_status, subset=["Status"]),
        use_container_width=True,
        height=400
    )

    st.caption(f"Menampilkan {len(laporan)} baris data dari {total_hari} hari unik.")

# HALAMAN VISUALISASI INTERAKTIF
elif menu == "📈 Visualisasi Data":

    st.title("📈 Visualisasi Interaktif Hotspot Kebakaran Hutan")

    df = load_map()

    # FILTER USER
    st.sidebar.subheader("🔍 Filter Visualisasi")

    # tanggal
    tgl_awal = st.sidebar.date_input(
        "Tanggal Awal",
        value=df["Tanggal"].min().date(),
        min_value=pd.to_datetime("2021-01-01").date(),
        max_value=pd.to_datetime("2025-12-31").date()
    )

    tgl_akhir = st.sidebar.date_input(
        "Tanggal Akhir",
        value=df["Tanggal"].max().date(),
        min_value=pd.to_datetime("2021-01-01").date(),
        max_value=pd.to_datetime("2025-12-31").date()
    )

    kabupaten = st.sidebar.multiselect(
        "Kabupaten",
        sorted(df["Kabupaten"].dropna().unique()),
        default=[]
    )

    confidence = st.sidebar.multiselect(
        "Confidence",
        sorted(df["Confidence"].astype(str).unique()),
        default=[]
    )

    # filter data
    df["Confidence"] = df["Confidence"].astype(str)

    df = df[
        (df["Tanggal"].dt.date >= tgl_awal) &
        (df["Tanggal"].dt.date <= tgl_akhir) &
        (df["Kabupaten"].isin(kabupaten)) &
        (df["Confidence"].isin(confidence))
    ]

    st.info(f"Data ditampilkan: {len(df)}")

    # LAPORAN RINGKAS / DETAIL DATA
    # Tambahkan setelah filter data

    st.markdown("## 📋 Laporan Data Hotspot")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Hotspot", len(df))
    col2.metric("Jumlah Kabupaten", df["Kabupaten"].nunique())
    col3.metric("Confidence Tertinggi", df["Confidence"].max())

    # DETAIL LAPORAN

    if len(df) > 0:

        # kabupaten terbanyak
        kab_terbanyak = df["Kabupaten"].value_counts().idxmax()
        jumlah_kab = df["Kabupaten"].value_counts().max()

        # confidence terbanyak
        conf_terbanyak = df["Confidence"].value_counts().idxmax()
        jumlah_conf = df["Confidence"].value_counts().max()

        # tanggal terbanyak
        tanggal_terbanyak = df["Tanggal"].dt.date.value_counts().idxmax()
        jumlah_tanggal = df["Tanggal"].dt.date.value_counts().max()

        # waktu terbanyak
        df_temp = df.copy()
        df_temp["Waktu_str"] = df_temp["Waktu"].astype(str).str.replace("WIB", "", regex=False).str.strip()
        df_temp["Jam"] = pd.to_numeric(df_temp["Waktu_str"].str.split(":").str[0], errors="coerce")
        data_jam_temp = df_temp.dropna(subset=["Jam"])
        data_jam_temp["Jam_2"] = (data_jam_temp["Jam"] // 2) * 2
        data_jam_temp["Rentang_Waktu"] = data_jam_temp["Jam_2"].apply(lambda x: f"{int(x):02d}:00 - {int(x)+1:02d}:59")
        if len(data_jam_temp) > 0:
            waktu_terbanyak = data_jam_temp["Rentang_Waktu"].value_counts().idxmax()
        else:
            waktu_terbanyak = "-"

        st.success(f"""
        📌 Analisis:

        • Total hotspot terdeteksi sebanyak **{len(df)} titik**  
        • Tersebar di **{df['Kabupaten'].nunique()} kabupaten**  
        • Kabupaten dengan hotspot terbanyak adalah **{kab_terbanyak}** sebanyak **{jumlah_kab} titik**  
        • Confidence yang paling sering muncul adalah **Level {conf_terbanyak}** sebanyak **{jumlah_conf} data**  
        • Tanggal kejadian tertinggi terjadi pada **{tanggal_terbanyak}** sebanyak **{jumlah_tanggal} hotspot**
        • Waktu kejadian hotspot terbanyak pada pukul **{waktu_terbanyak}**
        """)

    # TABEL DETAIL

    df_tampil = df.copy()
    df_tampil["Tanggal"] = df_tampil["Tanggal"].apply(format_tanggal_indo)

    st.dataframe(
    df_tampil[
        [
            "Tanggal",
            "Waktu",
            "Kabupaten",
            "Confidence",
            "Latitude",
            "Longitude"
        ]
    ],
    use_container_width=True
    )
    
    if len(df) == 0:
        st.warning("Silakan pilih Kabupaten dan Confidence terlebih dahulu")
    else:
        st.subheader("📊 Semua Visualisasi")

        # Siapkan data untuk Waktu Kejadian
        df["Waktu_str"] = (
            df["Waktu"]
            .astype(str)
            .str.replace("WIB", "", regex=False)
            .str.strip()
        )
        df["Jam"] = df["Waktu_str"].str.split(":").str[0]
        df["Jam"] = pd.to_numeric(df["Jam"], errors="coerce")
        data_jam = df.dropna(subset=["Jam"]).copy()
        data_jam["Jam_2"] = (data_jam["Jam"] // 2) * 2
        data_jam["Rentang_Waktu"] = data_jam["Jam_2"].apply(
            lambda x: f"{int(x):02d}:00 - {int(x)+1:02d}:59"
        )
        urutan = [f"{i:02d}:00 - {i+1:02d}:59" for i in range(0,24,2)]

        UKURAN = (5.5, 3.2)
        FONTSIZE = 7

        # BARIS 1
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=UKURAN)
            df["Kabupaten"].value_counts().head(10).plot(kind="bar", color="tomato", ax=ax)
            ax.set_xlabel("Kabupaten", fontsize=FONTSIZE)
            ax.set_ylabel("Jumlah Hotspot", fontsize=FONTSIZE)
            plt.xticks(rotation=45, fontsize=FONTSIZE)
            ax.tick_params(axis='both', labelsize=FONTSIZE)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=UKURAN)
            df["Tanggal"].value_counts().sort_index().plot(marker="o", ax=ax)
            ax.set_xlabel("Tanggal", fontsize=FONTSIZE)
            ax.set_ylabel("Jumlah Hotspot", fontsize=FONTSIZE)
            ax.tick_params(axis='both', labelsize=FONTSIZE)
            st.pyplot(fig)

        # BARIS 2
        col3, col4 = st.columns(2)

        with col3:
            fig, ax = plt.subplots(figsize=UKURAN)
            sns.countplot(data=df, x="Confidence", palette="Set2", ax=ax)
            ax.tick_params(axis='both', labelsize=FONTSIZE)
            st.pyplot(fig)

        with col4:
            fig, ax = plt.subplots(figsize=UKURAN)
            sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="Confidence", palette="Set1", ax=ax)
            ax.tick_params(axis='both', labelsize=FONTSIZE)
            ax.legend(fontsize=FONTSIZE)
            st.pyplot(fig)

        # BARIS 3
        fig, ax = plt.subplots(figsize=UKURAN)
        sns.countplot(data=data_jam, x="Rentang_Waktu", order=urutan, color="steelblue", ax=ax)
        plt.xticks(rotation=45, fontsize=FONTSIZE)
        ax.tick_params(axis='both', labelsize=FONTSIZE)
        st.pyplot(fig)

    # # HEATMAP
    # elif grafik == "Heatmap Korelasi":

    #     st.subheader("📌 Korelasi Data Numerik")

    #     num = df[["Latitude", "Longitude"]].corr()

    #     fig, ax = plt.subplots(figsize=(8,5))

    #     sns.heatmap(
    #         num,
    #         annot=True,
    #         cmap="coolwarm",
    #         ax=ax
    #     )

    #     st.pyplot(fig)

# HALAMAN CLUSTERING
elif menu == "📊 Hasil Clustering":

    st.title("📊 Dashboard Hasil Clustering")

    df = load_cluster()
    model, scaler = load_model()

    # filter cluster
    cluster_filter = st.sidebar.multiselect(
        "Pilih Cluster",
        sorted(df["Cluster"].unique()),
        default=sorted(df["Cluster"].unique())
    )

    df = df[df["Cluster"].isin(cluster_filter)]

    # KPI
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Jumlah Kabupaten", len(df))
    c2.metric("Total Hotspot", int(df["Jumlah_Hotspot"].sum()))
    c3.metric("Cluster Aktif", df["Cluster"].nunique())

    if "Total_Luas_Ha" in df.columns:
        c4.metric("Total Luas (Ha)", f"{df['Total_Luas_Ha'].sum():.1f}")
        c5.metric("Rata Luas (Ha)", f"{df['Rata_Luas_Ha'].mean():.1f}")
    else:
        c4.metric("Total Luas (Ha)", "-")
        c5.metric("Rata Luas (Ha)", "-")

    # tabel
    st.subheader("📄 Data Clustering")
    kolom_tampil = ["Kabupaten", "Cluster", "Jumlah_Hotspot", "Rata_Confidence"]
    if "Total_Luas_Ha" in df.columns:
        kolom_tampil += ["Total_Luas_Ha", "Jumlah_Kejadian", "Rata_Luas_Ha"]
    st.dataframe(df[kolom_tampil], use_container_width=True)

    UKURAN = (5.5, 3.2)
    FONTSIZE = 7

    # barplot hotspot
    st.subheader("📊 Hotspot per Kabupaten")

    fig1, ax1 = plt.subplots(figsize=UKURAN)

    sns.barplot(
        data=df.sort_values("Jumlah_Hotspot", ascending=False),
        x="Kabupaten",
        y="Jumlah_Hotspot",
        hue="Cluster",
        dodge=False,
        palette="Set1",
        ax=ax1
    )

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, fontsize=FONTSIZE)
    ax1.tick_params(axis='both', labelsize=FONTSIZE)
    ax1.set_xlabel("Kabupaten", fontsize=FONTSIZE)
    ax1.set_ylabel("Jumlah_Hotspot", fontsize=FONTSIZE)

    st.pyplot(fig1)

    # barplot luas kebakaran aktual
    if "Total_Luas_Ha" in df.columns:
        st.subheader("🔥 Luas Kebakaran Aktual per Kabupaten")

        fig_luas, ax_luas = plt.subplots(figsize=UKURAN)

        sns.barplot(
            data=df.sort_values("Total_Luas_Ha", ascending=False),
            x="Kabupaten",
            y="Total_Luas_Ha",
            hue="Cluster",
            dodge=False,
            palette="Set1",
            ax=ax_luas
        )

        ax_luas.set_xticklabels(ax_luas.get_xticklabels(), rotation=90, fontsize=FONTSIZE)
        ax_luas.tick_params(axis='both', labelsize=FONTSIZE)
        ax_luas.set_xlabel("Kabupaten", fontsize=FONTSIZE)
        ax_luas.set_ylabel("Total Luas (Ha)", fontsize=FONTSIZE)

        st.pyplot(fig_luas)

        st.subheader("📋 Jumlah Kejadian Kebakaran per Kabupaten")

        fig_jml, ax_jml = plt.subplots(figsize=UKURAN)

        sns.barplot(
            data=df.sort_values("Jumlah_Kejadian", ascending=False),
            x="Kabupaten",
            y="Jumlah_Kejadian",
            hue="Cluster",
            dodge=False,
            palette="Set1",
            ax=ax_jml
        )

        ax_jml.set_xticklabels(ax_jml.get_xticklabels(), rotation=90, fontsize=FONTSIZE)
        ax_jml.tick_params(axis='both', labelsize=FONTSIZE)
        ax_jml.set_xlabel("Kabupaten", fontsize=FONTSIZE)
        ax_jml.set_ylabel("Jumlah Kejadian", fontsize=FONTSIZE)

        st.pyplot(fig_jml)

    # scatter
    st.subheader("📍 Sebaran Cluster")

    fig2, ax2 = plt.subplots(figsize=UKURAN)

    sns.scatterplot(
        data=df,
        x="Longitude",
        y="Latitude",
        hue="Cluster",
        size="Jumlah_Hotspot",
        sizes=(80,400),
        palette="Set1",
        ax=ax2
    )

    ax2.tick_params(axis='both', labelsize=FONTSIZE)
    ax2.legend(fontsize=FONTSIZE)

    st.pyplot(fig2)

    # DENDROGRAM
    st.subheader("🌲 Dendrogram Hierarchical Clustering")

    fitur_dendo = ["Jumlah_Hotspot", "Rata_Confidence", "Latitude", "Longitude"]
    if "Total_Luas_Ha" in df.columns:
        fitur_dendo += ["Total_Luas_Ha", "Jumlah_Kejadian", "Rata_Luas_Ha"]

    X_dendo = df[fitur_dendo].values
    scaler_dendo = StandardScaler()
    X_scaled_dendo = scaler_dendo.fit_transform(X_dendo)

    Z = linkage(X_scaled_dendo, method="ward")

    fig_dendo, ax_dendo = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z,
        labels=df["Kabupaten"].values,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax_dendo
    )
    ax_dendo.set_title("Dendrogram Hierarchical Clustering (Ward)")
    ax_dendo.set_xlabel("Kabupaten")
    ax_dendo.set_ylabel("Jarak")
    fig_dendo.tight_layout()
    st.pyplot(fig_dendo)

# HALAMAN FORECASTING & ANOMALI
elif menu == "🔮 Forecasting & Anomali":

    st.title("🔮 Time Series Forecasting & Anomali Detection")

    tab1, tab2 = st.tabs(["📈 Forecasting Hotspot", "⚠️ Anomali Detection"])

    with tab1:
        st.markdown("""
        **SARIMA Model** — memprediksi jumlah hotspot bulanan berdasarkan data historis 2021-2025.
        """)

        try:
            df_hist = pd.read_csv("histori_hotspot.csv")
            df_forecast = pd.read_csv("forecast_hotspot.csv")

            df_hist["Tanggal"] = pd.to_datetime(df_hist["Tanggal"])
            df_forecast["Tanggal"] = pd.to_datetime(df_forecast["Tanggal"])

            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(df_hist["Tanggal"], df_hist["Jumlah_Hotspot"],
                    label="Historis", color="blue", linewidth=2)
            ax.plot(df_forecast["Tanggal"], df_forecast["Prediksi"],
                    label="Prediksi", color="orange", linestyle="--", linewidth=2)
            ax.fill_between(df_forecast["Tanggal"],
                            df_forecast["CI_Bawah"], df_forecast["CI_Atas"],
                            color="orange", alpha=0.2, label="Interval Kepercayaan 95%")

            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Jumlah Hotspot")
            ax.set_title("Time Series Forecasting Hotspot Kebakaran Jawa Timur (SARIMA)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

            st.subheader("📋 Tabel Prediksi 12 Bulan")
            df_tabel = df_forecast.copy()
            df_tabel["Tanggal"] = df_tabel["Tanggal"].dt.strftime("%B %Y")
            df_tabel.columns = ["Bulan", "Prediksi Hotspot", "Batas Bawah", "Batas Atas"]
            df_tabel[["Prediksi Hotspot", "Batas Bawah", "Batas Atas"]] = \
                df_tabel[["Prediksi Hotspot", "Batas Bawah", "Batas Atas"]].round(0).fillna(0).astype(int)
            st.dataframe(df_tabel, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Rata-rata Historis/Bulan",
                      f"{df_hist['Jumlah_Hotspot'].mean():.0f}")
            c2.metric("Rata-rata Prediksi/Bulan",
                      f"{df_forecast['Prediksi'].mean():.0f}")

        except FileNotFoundError:
            st.warning("Data forecasting belum tersedia. Jalankan notebook terlebih dahulu (cell Time Series Forecasting).")
            st.info("Atau klik tombol di bawah untuk generate sekarang:")

            if st.button("Generate Forecasting Sekarang"):
                with st.spinner("Menghitung forecasting..."):
                    from statsmodels.tsa.statespace.sarimax import SARIMAX

                    df = load_map()
                    df_bulanan = df.set_index("Tanggal").resample("ME").size().reset_index(name="Jumlah_Hotspot")
                    df_bulanan = df_bulanan[df_bulanan["Tanggal"] >= "2021-01-01"].set_index("Tanggal")

                    model = SARIMAX(df_bulanan["Jumlah_Hotspot"], order=(1,1,1),
                                    seasonal_order=(1,1,1,12),
                                    enforce_stationarity=False, enforce_invertibility=False)
                    result = model.fit(disp=False)
                    forecast = result.get_forecast(steps=12)
                    forecast_mean = forecast.predicted_mean
                    forecast_ci = forecast.conf_int()

                    last_date = df_bulanan.index[-1]
                    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="ME")

                    df_forecast = pd.DataFrame({
                        "Tanggal": forecast_index,
                        "Prediksi": forecast_mean.values,
                        "CI_Bawah": forecast_ci.iloc[:, 0].values,
                        "CI_Atas": forecast_ci.iloc[:, 1].values
                    })
                    df_forecast.to_csv("forecast_hotspot.csv", index=False)

                    df_hist = df_bulanan.reset_index()
                    df_hist.columns = ["Tanggal", "Jumlah_Hotspot"]
                    df_hist.to_csv("histori_hotspot.csv", index=False)

                    st.success("Forecasting berhasil! Refresh halaman untuk melihat hasil.")
                    st.rerun()

    with tab2:
        st.markdown("""
        **Z-Score Method** — mendeteksi bulan-bulan dengan jumlah hotspot yang tidak wajar (di luar 3 standar deviasi dari rata-rata per kabupaten).
        """)

        try:
            df_anomali = pd.read_csv("anomali_hotspot.csv")
            if "Kabupaten" not in df_anomali.columns:
                raise FileNotFoundError("Kolom Kabupaten tidak ditemukan")
            df_anomali["Anomali"] = df_anomali["Anomali"].astype(bool)

            total_anomali = df_anomali[df_anomali["Anomali"] == True]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Bulan-Kabupaten", len(df_anomali))
            c2.metric("Total Anomali", len(total_anomali))
            c3.metric("Persentase Anomali",
                      f"{len(total_anomali)/len(df_anomali)*100:.2f}%" if len(df_anomali) > 0 else "0%")

            # Filter kabupaten
            kab_list = sorted(df_anomali["Kabupaten"].unique())
            pilih_kab = st.multiselect("Filter Kabupaten", kab_list, default=[])

            df_tampil = df_anomali.copy()
            if pilih_kab:
                df_tampil = df_tampil[df_tampil["Kabupaten"].isin(pilih_kab)]

            # Tabel anomali
            df_anomali_only = df_tampil[df_tampil["Anomali"] == True].copy()
            if len(df_anomali_only) > 0:
                st.subheader("⚠️ Daftar Anomali Terdeteksi")
                df_anomali_only["Bulan_Teks"] = df_anomali_only.apply(
                    lambda r: f"{r['Tahun']}-{int(r['Bulan']):02d}", axis=1
                )
                st.dataframe(
                    df_anomali_only[["Kabupaten", "Bulan_Teks", "Jumlah_Hotspot", "Z_Score"]]
                    .rename(columns={
                        "Bulan_Teks": "Periode",
                        "Jumlah_Hotspot": "Hotspot",
                        "Z_Score": "Z-Score"
                    })
                    .sort_values("Z-Score", ascending=False),
                    use_container_width=True
                )

                # Barplot jumlah anomali per kabupaten
                st.subheader("📊 Jumlah Anomali per Kabupaten")
                fig, ax = plt.subplots(figsize=(12, 5))
                anomali_count = total_anomali["Kabupaten"].value_counts()
                sns.barplot(x=anomali_count.index, y=anomali_count.values, color="red", alpha=0.7, ax=ax)
                ax.set_xlabel("Kabupaten")
                ax.set_ylabel("Jumlah Bulan Anomali")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.success("Tidak ada anomali terdeteksi dengan threshold Z-Score > 3.")

            with st.expander("📋 Semua Data (termasuk non-anomali)"):
                st.dataframe(
                    df_tampil.sort_values(["Kabupaten", "Tahun", "Bulan"]),
                    use_container_width=True
                )

        except FileNotFoundError:
            st.warning("Data anomali belum tersedia. Jalankan notebook terlebih dahulu (cell Anomali Detection).")
            if st.button("Generate Anomali Detection Sekarang"):
                with st.spinner("Menghitung anomali..."):
                    df = load_map()
                    df["Tahun"] = df["Tanggal"].dt.year
                    df["Bulan"] = df["Tanggal"].dt.month

                    df_bulanan_kab = df.groupby(["Kabupaten", "Tahun", "Bulan"]).size().reset_index(name="Jumlah_Hotspot")

                    def deteksi_anomali(group):
                        mean = group["Jumlah_Hotspot"].mean()
                        std = group["Jumlah_Hotspot"].std()
                        if std == 0:
                            group["Anomali"] = False
                            group["Z_Score"] = 0.0
                        else:
                            group["Z_Score"] = (group["Jumlah_Hotspot"] - mean) / std
                            group["Anomali"] = group["Z_Score"] > 3
                        return group

                    df_anomali = df_bulanan_kab.groupby("Kabupaten", group_keys=False).apply(deteksi_anomali)
                    df_anomali.to_csv("anomali_hotspot.csv", index=False)
                    st.success("Anomali detection berhasil! Refresh halaman untuk melihat hasil.")
                    st.rerun()

# FOOTER
st.success("Terima Kasih")
