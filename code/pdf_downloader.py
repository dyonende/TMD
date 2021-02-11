'''
@authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
download pdf files of articles
'''
import pandas as pd
import requests
import argparse
import sys
import os

def check_path(input_file, output_path):
    if os.path.isfile(input_file) == False:
        print(f"{input_file} is not an existing file")

    if output_path[:-1] != '/':
        output_path+'/'
        
    if os.path.isdir(output_path) == False:
        print(f"{output_path} is not an existing directory")
        sys.exit()

def collect_links(input_file):
    '''
    collect the id and url to article for each entry in the data
    '''
    with open(input_file) as infile:
        df = pd.read_csv(infile, delimiter=',')
        ids = df['_id']
        links = df['unpaywall_response.free_fulltext_url'] #column with urls
        if len(links)==len(ids):
            return ids, links
            
def download_pdf(ids, links, output_path):
    '''
    try to download the pdf at the specified url and saved 
    it to a file with the id as name
    '''
    number_of_links = len(links)
    for i in range(number_of_links):
        print(str(i/number_of_links*100)[:4]+"%", end="\r") #print progress
        id = ids[i]
        link = links[i]
        filename = id+'.pdf'
        try:            
            r = requests.get(link, stream=True) 
        except KeyboardInterrupt:
            sys.exit()
        except:
            print(id +" failed to download")
        
        with open(output_path+filename, 'wb') as outfile:
                outfile.write(r.content)
          

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='file path to data')
    parser.add_argument('output_path',
                        help='file path to store pdf\'s')
    args = parser.parse_args()
    input_file = args.input_file
    output_path = args.output_path
    
    check_path(input_file, output_path)

    ids, links = collect_links(input_file)
    download_pdf(ids, links, output_path)

if __name__ == '__main__':
    main()
