##Apriori Tensorflow

Apriori implementation using tensorflow backend

#### TODOs
1. Test with a large sample set to check performance against e.g. apyori 
 * This shows the memory usage is a big issue - even 100'000 inputs with 30 possible values 
 surpassed 100Gb (95Gb compressed) usage. Need to remove rows which are no longer possible.
 The lack of sparse element wise multiplication is now a blocker to this project. 
2. Neaten up output - return AprioriFrequentSets
3. Put apriori.py functions into classes, store result as member for further processing
4. Add feature to return list of all 'members' (input row ids) in each frequent group
5. Add feature to calculate distance metric between groups
