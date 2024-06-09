!pip install torch-geometric
!pip install torch-scatter
!pip install torch-sparse
!pip install torch-cluster
!pip install torch-spline-conv
!pip install py2neo

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from py2neo import Graph, Node, Relationship

# Neo4j Aura'ya bağlanma
uri = ""
username = ""
password = ""

graph = Graph(uri, auth=(username, password))

# Veri setini yükleyelim
file_path = '/mnt/data/corrected_heart_data.csv'
df = pd.read_csv(file_path)
df.head()

# Hedef değişken ve özellikler
target_column = 'HeartDisease'  # Hedef değişkenin doğru adını buraya yazın
X = df.drop(columns=[target_column])
y = df[target_column]

# Hedef değişkeni 0 tabanlı hale getirme
y = y - 1

# Veri türlerini kontrol edip uygun biçime dönüştürme
def prepare_data(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype(float)
    return df

df = prepare_data(df)

# Hata ayıklama çıktıları ekleme
def load_data_to_neo4j(df, graph, chunk_size=50):
    graph.run("MATCH (n) DETACH DELETE n")  # Var olan tüm düğümleri ve ilişkileri silme

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]
        for index, row in chunk.iterrows():
            try:
                patient = Node("Patient", id=index, **row.to_dict())
                graph.create(patient)
                print(f"Successfully created node for index {index}")
            except Exception as e:
                print(f"Failed to create node for index {index} with error: {e}")

# Veri setini yükleme
load_data_to_neo4j(df, graph)

# Neo4j'den veri çekme
def fetch_data_from_neo4j(graph):
    query = """
    MATCH (p:Patient)
    RETURN p
    """
    data = graph.run(query).to_data_frame()
    return data
    # Veriyi çekme
neo4j_data = fetch_data_from_neo4j(graph)

# Veri çerçevesindeki sütun adlarını kontrol etme
print(neo4j_data.columns)

# Sütun adlarını düzeltme
neo4j_data = pd.json_normalize(neo4j_data['p'])
print(neo4j_data.columns)

# Doğru sütun adını bulduktan sonra kodu devam ettirin
correct_target_column = 'HeartDisease'  # Bu adı sütun adlarına göre güncelleyin

# İlişkileri oluşturma
def create_relationships(graph):
    query = """
    MATCH (a:Patient), (b:Patient)
    WHERE a.HeartDisease = b.HeartDisease AND a.id < b.id
    CREATE (a)-[:SIMILAR]->(b)
    """
    graph.run(query)

# İlişkileri oluştur
create_relationships(graph)

# Veriyi hazırlama (PyTorch Geometric formatı için)
features = torch.tensor(neo4j_data.drop(columns=[correct_target_column]).values, dtype=torch.float)
labels = torch.tensor(neo4j_data[correct_target_column].values, dtype=torch.long)

# veri setini train ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardizasyon (ortalamalarını 0, standart sapmasını 1 yapar)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hedef değişkenin sınıf dağılımını kontrol edelim
print(y_train.unique())

batch_size = 32

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                         torch.tensor(y_train.values, dtype=torch.long)),
                          batch_size=batch_size, shuffle=True)

test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                        torch.tensor(y_test.values, dtype=torch.long)),
                         batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.dropout = nn.Dropout(dropout_prob)  # dropout katmanı
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # dropout uygulanması
        x = self.fc4(x)
        return x

input_dim = X_train_scaled.shape[1]
hidden_dim1 = 256  # ilk gizli katman
hidden_dim2 = 128  # ikinci gizli katman
hidden_dim3 = 64   # üçüncü gizli katman
output_dim = 2     # binary classification için 2 sınıf
dropout_prob = 0.3 # dropout olasılığı

model = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout_prob)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 150

# Eğitim ve Test fonksiyonları
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy

def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    y_test_proba = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            probabilities = F.softmax(output, dim=1)
            y_test_proba.extend(probabilities[:, 1].tolist())
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy, y_test_proba

train_losses = []
train_accuracies = []
test_accuracies = []
y_test_proba = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    test_loss, test_acc, test_proba = test(model, test_loader, criterion)
    test_accuracies.append(test_acc)
    y_test_proba = test_proba  # Bu satırda y_test_proba güncellenecek

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')

# Eğitim ve Test sonuçlarını görselleştirme
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Eğitim Kaybı Grafiği
ax1.plot(range(num_epochs), train_losses, label='Training Loss', color='orange')
ax1.set_ylim(0, 0.7)
ax1.set_title('Training Loss over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Eğitim Doğruluğu Grafiği
ax2.plot(range(num_epochs), train_accuracies, label='Training Accuracy', color='yellow')
ax2.set_ylim(0.55, 1)
ax2.set_title('Accuracy over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()

# ROC Curve ve AUC Grafiği
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Kullanıcıdan alınan verilerle tahmin yapma
def predict(user_data):
    model.eval()
    user_data = torch.tensor(user_data, dtype=torch.float).unsqueeze(0)
    output = model(user_data)
    probabilities = F.softmax(output, dim=1)
    prob = probabilities.squeeze().tolist()
    return prob

# Kullanıcıdan veri alma ve tahmin yapma
def get_user_input():
    age = float(input("Yaşınızı girin: "))
    sex = float(input("Cinsiyet (0: Kadın, 1: Erkek): "))
    cp = float(input("Göğüs ağrısı tipi (1-4): "))
    trestbps = float(input("Dinlenme kan basıncı: "))
    chol = float(input("Kolesterol: "))
    fbs = float(input("Açlık kan şekeri > 120 mg/dl (1: Evet, 0: Hayır): "))
    restecg = float(input("Dinlenme elektrokardiyografik sonuçları (0-2): "))
    thalach = float(input("Maksimum kalp hızı: "))
    exang = float(input("Egzersiz indüklenmiş anjina (1: Evet, 0: Hayır): "))
    oldpeak = float(input("ST depresyonu: "))
    slope = float(input("ST segmentinin eğimi: "))
    ca = float(input("Büyük damarların sayısı (0-3): "))
    thal = float(input("Talasemi (3: Normal, 6: Sabit Kusur, 7: Geri Dönüşümlü Kusur): "))

    user_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    user_data_df = pd.DataFrame([user_data], columns=X.columns)  # Veri setine uygun özellik adlarıyla DataFrame oluşturma
    user_data_scaled = scaler.transform(user_data_df)  # Veriyi normalize etme
    return user_data_scaled[0]

# Kullanıcı verisini al ve tahmin yap
user_data = get_user_input()
risk_probs = predict(user_data)
risk_percentage = risk_probs[1] * 100
print(f'Kalp krizi riski: {risk_percentage:.2f}%')

# Kullanıcının kalp krizi riskini grafikle gösterme
plt.figure(figsize=(6, 4))
plt.bar(['Kalp Krizi Riski'], [risk_percentage], color='orange')
plt.ylim(0, 100)
plt.ylabel('Risk Yüzdesi')
plt.title('Kullanıcının Kalp Krizi Riski')
plt.show()