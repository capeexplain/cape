import sys
from capexplain.explain.explanation import main as capemain

def main(argv=[]):
    capemain(sys.argv[1:])

if __name__=='__main__':
    main()
