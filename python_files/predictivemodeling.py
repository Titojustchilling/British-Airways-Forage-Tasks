import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/customer_booking.csv', encoding="ISO-8859-1")
X = data.drop('booking_complete',axis=1)
Y = data['booking_complete']        
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

classify = RandomForestClassifier(n_estimators=100, random_state=0)

classify.fit(X_train, Y_train)
y_pred = classify.predict(X_test) 
imp_features = classify.feature_importances_
imp_features = pd.Series(imp_features, name="Important Features", index=X.columns)
imp_features = imp_features.sort_values(ascending=True)



def importance_scores_graph(scores): 
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores, color='green')
    plt.yticks(width, ticks)
    plt.title("Features Ranked By Their Importance")



#plt.figure(figsize=(15,15))
plt.title("Features Ranked By Their Importance")
#plt.figure(dpi=100, figsize=(8, 5))
importance_scores_graph(imp_features)
plt.show()

print("ACCURACY: ", metrics.accuracy_score(Y_test, y_pred))




