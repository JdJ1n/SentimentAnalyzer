def bert_classifier(batch_size, dr):
    import numpy as np
    import pandas as pd
    import torch
    import transformers as ppb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import warnings
    warnings.filterwarnings('ignore')

    # Use the DataReader to read the data
    data, labels = dr.get_labelled_data()

    # Convert the data and labels to a DataFrame
    df = pd.DataFrame({
        'data': data,
        'labels': labels
    })

    # Set up BERT tokenizer and model
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Tokenize the text
    tokenized = df['data'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # Pad the tokenized data
    max_len = max(len(i) for i in tokenized.values)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)

    # Convert to PyTorch tensors
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    # Initialize an empty list to hold the BERT output
    result = []

    # Process the input data in batches
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], batch_size):
            last_hidden_states_for_batch = model(input_ids[i: i + batch_size],
                                                 attention_mask=attention_mask[i: i + batch_size])
            result.append(last_hidden_states_for_batch[0])

    # Concatenate the results
    last_hidden_states = torch.cat(result, dim=0)

    # Extract the features (the output of the [CLS] tokens)
    features = last_hidden_states[:, 0, :].numpy()

    # Manually split the dataset into training and testing
    split_idx = int(len(features) * 0.7)
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    # Initialize and train the logistic regression classifier
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    # Evaluate the classifier
    acc = lr_clf.score(test_features, test_labels)
    pred = lr_clf.predict(test_features)
    f1 = f1_score(test_labels, pred, average='weighted')

    return acc, f1
