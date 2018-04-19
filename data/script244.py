
# coding: utf-8

# Below a discussion of team Waffle convolutions inc.'s implementation of the 1D convolutional model (place 45 on the leaderboard), show our biased comparison to LSTMs, quickly describe its performance and explain some of our 'design' choices. 
# 
# ### Comparison to LSTM
# 
# A quick overview of its benefits with respect to the LSTM approach:
# 
# - Doesn't overfit as much
# 
# - Much shorter run times, 1 epoch ~ 130 sec on a 1050 GPU, as opposed to 400 sec epochs for a 250 unit LSTM
# 
# - Intuitively much clearer what is happening (might just be me though)
# 
# - Seems to perform better (with only magic features, 0.160 single model on lb) 
# 
# Downsides of the 1D CNN:
# 
# - Sentences which are 'equal' but have different sequence of words (eg. 'Is bacon the best thing since sliced bread?' or 'Since sliced bread is bacon the best thing?') flunk easily  
# 
# - No concept of word importance (such TFIDF, can be added with a dense layer though) 
# 
# - Our implementation cannot switch question1 with question2 to double the data
# 
# However in all fairness, we played around more with the 1D convolutional models than LSTMs, we simply couldn't get the latter to have similar performance.
# 
# ### The actual model

# In[ ]:


def model_conv1D_(emb_matrix):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=60,
        trainable=False
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(60,))
    seq2 = Input(shape=(60,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    # Add the magic features
    magic_input = Input(shape=(5,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
    # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    distance_input = Input(shape=(20,))
    distance_dense = BatchNormalization()(distance_input)
    distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer
    merge = concatenate([diff, mul, magic_dense, distance_dense])

    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


# ### Performance
# 
# All scores mentioned are from the public leaderboard. To compare it with your own stuff:
# 
# - Clean CNN, so we don't add external features. **0.31** without class weights **0.22** with class weights
# 
# - CNN + magic features, we used the frequency of q1 and q2, intersection of the questions and the maxkcore of q1 and 2 (didn't seem to make difference, besides faster convergence).  **0.156**
# 
# - CNN + magic and regular features, we added tfidf over character and word ngrams, tfidf in combination with wordmatch, the length of q1 and q2, the number of words the questions had in common, fuzzy string matching, word mover distance and skew and kurtosis of q1 and q2. **0.141**
# 
# - Ensemble of CNNs, using geometric averaging over 15 models gave us a score of **0.133**
# 
# ### 'Design' Choices
# 
# So basically why did we do certain stuff, this should be the more educational part (also for us, if you think we are doing something wrong please say so!). 
# 
# - Multiple convolutional layers with different kernel sizes, so the size of the kernel is the number of words your filters see. So basically a 1D convolution with kernel size 3 can be understood as a very fancy 3-gram of words. By simply adding more convolutional layers with a different size you obtain different grams which should tell you something new. We noticed that kernel sizes > 6 didn't contribute, but did help in overfitting so we stopped there
# 
# - GlobalAveragePooling (GAP) layer instead of max pooling,  two reasons, it was faster since we didn't need to flatten the output and GAP performed better (we suspect that GAP preserves more information than maxpooling)
# 
# - The absolute difference and the multiplication layer for parsing the convolutional/GAP output, since the questions are passed through the same convolutional layer they are represented in the same way. The only thing that is important for determining whether the questions are different is actual the is the difference between the questions. So we take the absolute difference and product between the corresponding 'output' nodes. 
# 
# - Why didnt we concatenate the two outputs (?), which is what is usually done by siamese networks. By taking the absolute difference and multiplying we are destroying information. Furthermore the dense layers after the concatenate can 'learn' to take the absolute difference and multiply the two questions.  *So we think (this is basically speculation), that we only destroy information the network can use to overfit. Since some questions appear > 10 times the network starts to recognize their representation, but if we take the difference this is no longer true! However this also takes away our ability to switch question1 and question2 to double our data ...*.
# 
# We go over the concatenate vs absolute difference and multiplication, since it is something we believe makes the network better (we cant conclusively say if it is actually better, since the convergence was also a lot faster. It might be that after 2 or 3 times the number of epochs the concatenate catches up) and was a point of discussion in the team.
# 
# Anyway, please don't take the above as absolute fact it is only what we think is happening.
# 
# Any questions or comments are appreciated.
