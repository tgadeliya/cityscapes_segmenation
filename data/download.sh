!wget --directory-prefix=data/  http://www.mimuw.edu.pl/~ciebie/cityscapes.tgz 
!tar -xzf data/cityscapes.tgz -C data/
!python data/cityscapes/check_close.py # Check files
!rm data/cityscapes/check_close.py
