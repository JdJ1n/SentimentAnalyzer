def bert_classifier(batch_size, dr):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import torch
    import transformers as ppb
    import warnings
    warnings.filterwarnings('ignore')

    # Use the DataReader to read the data
    data, labels = dr.get_labelled_data()

    # Convert the data and labels to a DataFrame
    df = pd.DataFrame({
        'data': data,
        'labels': labels
    })

    # Want BERT instead of distilBERT? Uncomment the following line:
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    tokenized = df['data'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    from tqdm import tqdm

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    result = []

    with torch.no_grad():
      for i in tqdm(range(0, input_ids.shape[0], batch_size)):
        last_hidden_states_for_batch = model(input_ids[i: i + batch_size], attention_mask=attention_mask[i : i + batch_size])
        result.append(last_hidden_states_for_batch)


    last_hidden_states = torch.cat(list(map(lambda x: x.last_hidden_state, result)), dim=0)

    features = last_hidden_states[:,0,:].numpy()

    labels = df['labels']

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    acc = lr_clf.score(test_features, test_labels)

    pred = lr_clf.predict(test_features)
    f1 = f1_score(test_labels, pred, average='weighted')

    return acc, f1
