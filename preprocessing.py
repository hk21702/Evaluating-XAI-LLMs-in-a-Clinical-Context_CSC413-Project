import pandas as pd
import argparse as ap

def create_parser():
    parser = ap.ArgumentParser(description='Preprocessing of the data')
    parser.add_argument('--input', type=str, help='Input folder')
    parser.add_argument('--output', type=str, help='Output foilder')
    return parser

def main():
    pass