import lib_sklearn

df=lib_sklearn.pd.read_csv('./fake_news.csv')

df.head()

df.tail()

df.isnull()

df.isnull().sum()

df.info()

df.label.value_counts()

i=df.label.value_counts()

fig = lib_sklearn.go.Figure(data=[go.Bar(
            x=['Real','Fake'], y=i,
            text=i,
            textposition='auto',
        )])

fig.show()

X_train,X_test,y_train,y_test=lib_sklearn.train_test_split(df['text'], df.label, test_size=0.2, random_state=7)

X_train

X_train.shape

y_train

y_train.shape

X_test.shape

y_test.shape

tfidf_vectorizer=lib_sklearn.TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)

pac=lib_sklearn.PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)

score=lib_sklearn.accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

lib_sklearn.confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

print('\n clasification report:\n',lib_sklearn.classification_report(y_test,y_pred))

ii=['Red-state leaders emboldened by Donald Trump’s presidential victory are not waiting for him to take office to advance far more conservative agendas at home. Idaho lawmakers want to allow school staff to carry concealed firearms without prior approval and parents to sue districts in library and curriculum disputes. Lawmakers in Oklahoma plan to further restrict abortion by limiting the emergency exceptions and to require the Ten Commandments to be displayed in public schools, while their counterparts in Arkansas are moving to create the felony offense of “vaccine harm,” which could make pharmaceutical companies or their executive officers potentially criminally liable.']

ii=tfidf_vectorizer.transform(ii)

y_pred=pac.predict(ii)

y_pred





















