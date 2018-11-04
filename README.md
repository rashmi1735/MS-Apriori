# MS-Apriori
Association Rule Mining with Multiple Support Thresholds using MSApriori Algorithm

An implementation of the MSApriori Algorithm for pattern mining of frequent item sets. Using multiple minimum item supports for different items, the algorithm helps detect rare item rules without generating any irrelevant rules. Different constraints were introduced to limit the output size of frequent item sets for efficient rule generation while meeting specific user requirements.

Keep the MSApriori.py, input file and the parameter file in the same directory

Open the command line

Go to the directory where MSApriori.py, input file and the parameter file are stored

Run the below command

python MSApriori.py -i inputfilename.txt -p parameterfilename.txt

e.g.:

python MSApriori.py -i sample-input.txt -p sample-parameter.txt

The results can be seen in file named "output.txt" in the same directory as other files
