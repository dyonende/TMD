'''
@authors: Dyon van der Ende, Eva den Uijl, Myrthe Buckens
extract text from pdfs
'''

import argparse
import sys
import os
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
import glob
from io import StringIO

def check_path(path):
    if os.path.isfile(path) == False and os.path.isdir(path) == False:
        print(f"{path} is not valid")
        sys.exit()
        
    if os.path.isdir(path) and path[:-1] != '/':
        path += '/'
    
    return path
    
def extract_text(in_path, out_path):
#https://towardsdatascience.com/pdf-text-extraction-in-python-5b6ab9e92dd
    files = glob.glob(in_path+'*.pdf')
    for i in range(len(files)):
        print(str(i/len(files)*100)[:4]+"%", end="\r")
        name = files[i]
        file_path = in_path+name
        output_string = StringIO()
        with open(file_path, 'rb') as infile: 
            parser = PDFParser(infile)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
        out_filename = out_path+os.path.basename(name).replace("pdf", "txt")
        with open(out_filename, 'w') as outfile:
            outfile.write(output_string.getvalue())
           
         
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        help='file path to data')
    parser.add_argument('output_path',
                        help='file path to output folder')

    args = parser.parse_args()
    data_path = check_path(args.data_path)
    output_path = check_path(args.output_path)
    extract_text(data_path, output_path)
    
    
if __name__ == '__main__':
    main()
