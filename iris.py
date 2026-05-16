import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#データ読み込み
iris = load_iris()
x,y = iris.data, iris.target

print("アヤメの品種：", iris.target_names)
print("データの形：", x.shape)

#データを学習用とテスト用に分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

df = pd.DataFrame(x, columns=iris.feature_names)
df.head()

print("x_trainの形:", x_train.shape)
print("x_testの形:", x_test.shape)
print("y_trainの形:", y_train.shape)
print("y_testの形:", y_test.shape)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"正解率：{accuracy * 100:.1f}%")