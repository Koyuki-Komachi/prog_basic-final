import numpy as np  
import matplotlib.pyplot as plt  

# データの読み込み  
data1 = np.loadtxt('dat1.txt')  
data2 = np.loadtxt('dat2.txt')  

# 観測信号行列の作成（2×N行列）  
X = np.vstack((data1, data2))  

# 平均を0に調整  
X = X - np.mean(X, axis=1, keepdims=True)  

# 白色化  
# 共分散行列の計算  
cov = np.cov(X)  
# 固有値分解  
eigenvalues, eigenvectors = np.linalg.eigh(cov)  
# 白色化行列の計算  
D_inv = np.diag(1.0 / np.sqrt(eigenvalues))  
V = eigenvectors.dot(D_inv).dot(eigenvectors.T)  
# データの白色化  
Z = V.dot(X)  

def fastICA(z, max_iter=1000, tol=1e-6):  
    m = z.shape[0]  
    w = np.random.rand(m)  
    w = w / np.sqrt(np.sum(w**2))  
    
    for i in range(max_iter):  
        w_old = w.copy()  
        # ICAの更新式  
        w = np.mean(z * (np.dot(w_old.T, z))**3, axis=1) - 3 * w_old  
        w = w / np.sqrt(np.sum(w**2))  
        
        if np.abs(np.abs(np.dot(w.T, w_old)) - 1) < tol:  
            break  
    
    return w  

# 2つの独立成分を抽出  
w1 = fastICA(Z)  
# 2つ目の成分は最初の成分と直交するように初期化  
w2 = np.random.rand(2)  
w2 -= np.dot(w2, w1) * w1  
w2 = w2 / np.sqrt(np.sum(w2**2))  
w2 = fastICA(Z)  

# 分離行列の作成  
W = np.vstack((w1, w2))  

# 独立成分の抽出  
S = np.dot(W, Z)  

# 元のスケールに戻す  
S = S * np.std(X, axis=1, keepdims=True)  

# 結果のプロット  
plt.figure(figsize=(12, 6))  

plt.subplot(2, 1, 1)  
plt.plot(S[0])  
plt.title('Separated Signal 1')  
plt.ylabel('Amplitude')  
plt.grid(True)  

plt.subplot(2, 1, 2)  
plt.plot(S[1])  
plt.title('Separated Signal 2')  
plt.xlabel('Sample')  
plt.ylabel('Amplitude')  
plt.grid(True)  

plt.tight_layout()  
plt.show()