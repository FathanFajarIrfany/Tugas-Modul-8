import pandas as pd
import numpy as np
import streamlit as st


def baca_file(file):
    file = pd.read_csv(file)
    return file

def buat_array(df):
    # kolom terakhir = target (y), kolom lainnya = fitur (kolom 'sampel' diabaikan)
    fitur = [k for k in df.columns[:-1] if k.lower() != 'sampel']
    x = df[fitur]
    x = np.array(x)
    x = np.column_stack([np.ones(len(x)), x])  # kolom 1 untuk w0

    y = df[df.columns[-1]]
    y = np.array(y)
    return x, y, fitur

def cramer(A, b):
    n = A.shape[0]
    det_A = np.linalg.det(A)

    w = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b              # ganti kolom ke-i dengan b
        w[i] = np.linalg.det(A_i) / det_A
    return w

def hitung_norm_equation(x, y):
    XtX = x.T @ x # perkalian matriks
    Xty = x.T @ y

    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError("Matriks XtX singular, metode Cramer tidak dapat digunakan")

    w = cramer(XtX, Xty)

    return w

def tampilan_hasil(x, y, w):
    XtX = x.T @ x 
    Xty = x.T @ y
    
    y_prediksi = x @ w

    rmse = np.sqrt(np.mean((y_prediksi - y) ** 2))
    r2 = 1 - (np.sum((y - y_prediksi) ** 2) / np.sum((y - np.mean(y)) ** 2))

    # jumlah galat = jumlah (y asli - y prediksi)
    jumlah_galat = round(np.sum(y - y_prediksi), 6) + 0.0

    tabel_perbandingan = pd.DataFrame({
        "y_asli":y,
        "y_prediksi":y_prediksi
    })

    return rmse, y_prediksi, r2, XtX, Xty, tabel_perbandingan, jumlah_galat

def hitung_input(x_input, w):
    y_prediksi = np.dot(x_input,w)
    return y_prediksi

# pembatas bagian UI streamlit
st.title('pertemuan 8')

file = st.file_uploader("silahkah masukan file csv anda")

if file :
    df = baca_file(file)
    x, y, fitur = buat_array(df)
    w = hitung_norm_equation(x, y)
    rmse, y_prediksi, r2, XtX, Xty, tabel_perbandingan, jumlah_galat = tampilan_hasil(x, y, w)
    w_tampil = np.round(w, 4) + 0.0   # pembulatan hanya untuk tampilan

    st.dataframe(df)

    
    col_1, col_2 = st.columns(2)
    with col_1:
        st.write("bobot didapat")
        st.write(w_tampil)
        st.dataframe(tabel_perbandingan)
        
    with col_2:
        st.write(f" ini hasil rmse {rmse}")
        st.write(f" ini hasil r2 {r2}")
        st.write(f" ini jumlah galat {jumlah_galat}")
        st.write(f" ini adalah A {XtX}")
        st.write(f" ini adalah hasil B {Xty}")
        persamaan = f"{w_tampil[0]}"
        for i in range(len(fitur)):
            persamaan += f" + {w_tampil[i+1]}X{i+1}"
        st.write(f"hasil persamaan anda y = {persamaan}")

        x_input = [1]
        for nama in fitur:
            x_input.append(st.number_input(f"masukan {nama}", value=float(df[nama].mean())))
        x_input = np.array(x_input)
        
        if st.button("tekan"):
            y_prediksi = hitung_input(x_input, w)
            st.write(f"hore prediksi model anda adalah {y_prediksi}")