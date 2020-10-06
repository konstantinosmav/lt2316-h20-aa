# LT2316 H20 Assignment A1

Name: Konstantinos Mavromatakis

Part 1

For part 1, I used os.walk to iterate through the root directory and its subdirectories to gain access to all xml files. If “Train” was contained in the full path of the filename, I used choice from  numpy.random to get “train” with a distribution of 0.8 and “development” with a distribution of 0.2. “Test” was assigned in the other cases. Then, I used ElementTree to parse through all sentences in each file. Tokenization was really problematic, I put a lot of thought into it and tried my hardest to separate the tokens the correct way. However, it did not always go the way I wanted. At first, I thought of using MWE tokenizer to which I added all entity names with .add_mwe so as to not split the multi-word expressions. However, when I ran it, it took so much time that I decided not to.  So to tokenize, I basically ended up using .split(‘ ’) on whitespace, which created tokens that were not really needed (e.g. ‘(’) but also split some entities wrongly (e.g.“antidepressants;”) in the cases where words were followed by a punctuation mark. To be able to keep track of the number of punctuation marks in every token of interest, I replaced the most frequently occurring punctuation signs with “_” and then i did str.count(‘_’) when I wanted to obtain the character offset. Other entities that were multi-word expressions were split up  “ARRR=1.33,”,”95%”, ”CI=1.02-1.74” and, as a result, their character offset in the data_df  might be rather unrelated to what we were given in the XML files. I decided to map each unique token to a unique integer for the data_df and I did the same for the labels- if a word was not one of the four labels I assigned it a 0. As I mentioned above, for the ner_df  I used the character offset that we were given in the xml files. When there was only one entity, I would append the character offset as it was to the list I kept for the ner_df. However, when there was a discontinuous or multi-word entity, I split the entities into two separate ones as we were told and appended both instances to my list_for_ner_df.
For get_y, I created the helper function get_labels, which returns a list of lists with each list containing the labels given to each token of each sentence. I padded them using tf.keras.preprocessing.sequence.pad_sequences, then I turned them into tensors with torch.tensor and saved them to the GPU.
For plot_split_ner_distribution, I decided to not include the non-ner tokens because they outnumber the ner tokens so greatly that they would not show in the plot if I kept them.

Here are the helper functions I made:

instantiate_df was the main function that created the data frames with all the columns that we were asked.
get_max_len: returns the max sample length of all sequences.
get_id_only : returns the integer that I mapped to tokens and labels
padding : pads all sentences to the max_sample_length
get_labels : returns a list of lists that contain the labels of each sentence(I assigned 0 to non-ner tokens).


Part 2

For feature extraction, I decided to use word embeddings. I first padded the sentences to get them all to the same word length to be able to create the word embeddings. Then, I iterated through all sentences and token ids of each sentence and made them into tensors, which I appended to a list that represents each sentence. After that, I stacked it and appended it to the all_feat list that would contain all sentences. When the iterations were done, I stacked the all_feat list into a tensor of tensors of tensors and returned it. In the end, I called the get_feat function for the three splits and saved the tensors to the GPU.

Bonus

For plot_sample_length_distribution, I made a histogram that displays the length of sentences on the X axis to the number of sentences in each split on the Y axis.

For plot_ner_per_sample_distribution, I made  a histogram that displays the number of ners on the X axis along with the number of sentences on the Y axis.

For plot_ner_cooccurence_venndiagram, I pip installed venn on mltgpu and plotted a venn diagram of  a dictionary that had the 4 ner_labels as keys and the set of sentences that contained each ner_label as values in the form of a list.

To get run.ipynb to run, I made some small changes to get_random_sample.

