pythondef combine_embeddings(emb1, emb2):
    distance = np.linalg.norm(emb1 - emb2)
    return np.concatenate([emb1, emb2, [distance]])

features = []
labels = []
for idx, row in df.iterrows():
    main_path = os.path.join(base_path, row['main_mark_image'])
    earlier_path = os.path.join(base_path, row['earlier_mark_image'])
    emb_main = get_embedding(main_path).numpy().flatten()
    emb_earlier = get_embedding(earlier_path).numpy().flatten()
    combined = combine_embeddings(emb_main, emb_earlier)
    features.append(combined)
    labels.append(1 if row['similarity'] == 'similar' else 0)

features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=36, random_state=42)

clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, predictions))
print("F1 Score :", f1_score(y_test, predictions))

