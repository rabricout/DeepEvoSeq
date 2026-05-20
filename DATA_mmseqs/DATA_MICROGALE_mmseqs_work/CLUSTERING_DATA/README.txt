# Informations for a1, adapt for different data
1. python cat_data.py    # concatenate individual fasta files
2. mkdir a1DB
3. cp src/* a1DB
4. cp a1.fa a1DB
5. cd a1DB
6. bash easy_cluster.sh a1.fa
7. python split_clusters.py
# Output cluster fasta files are in CLUSTERED_DATA