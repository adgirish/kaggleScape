
# coding: utf-8

# # A Whirlwind Tour of Python
# 
# *Jake VanderPlas, Summer 2016*
# 
# These are the Jupyter Notebooks behind my O'Reilly report,
# [*A Whirlwind Tour of Python*](http://www.oreilly.com/programming/free/a-whirlwind-tour-of-python.csp).
# The full notebook listing is available [on Github](https://github.com/jakevdp/WhirlwindTourOfPython).
# 
# *A Whirlwind Tour of Python* is a fast-paced introduction to essential
# components of the Python language for researchers and developers who are
# already familiar with programming in another language.
# 
# The material is particularly aimed at those who wish to use Python for data 
# science and/or scientific programming, and in this capacity serves as an
# introduction to my upcoming book, *[The Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do)*.
# These notebooks are adapted from lectures and workshops I've given on these
# topics at University of Washington and at various conferences, meetings, and
# workshops around the world.

# ## Index
# 
# 1. [Introduction](https://www.kaggle.com/sohier/whirlwind-tour-of-python-00-introduction)
# 2. [How to Run Python Code](https://www.kaggle.com/sohier/wtop-01-how-to-run-python-code)
# 3. [Basic Python Syntax](https://www.kaggle.com/sohier/wtop-02-syntax)
# 4. [Python Semantics: Variables](https://www.kaggle.com/sohier/wtop-03-semantics-and-variables)
# 5. [Python Semantics: Operators](https://www.kaggle.com/sohier/wtop-04-semantics-and-operators)
# 6. [Built-In Scalar Types](https://www.kaggle.com/sohier/wtop-05-scalar-types)
# 7. [Built-In Data Structures](https://www.kaggle.com/sohier/wtop-06-built-in-data-structures)
# 8. [Control Flow Statements](https://www.kaggle.com/sohier/wtop-07-control-flow)
# 9. [Defining Functions](https://www.kaggle.com/sohier/wtop-08-functions)
# 10. [Errors and Exceptions](https://www.kaggle.com/sohier/wtop-09-errors-and-exceptions)
# 11. [Iterators](https://www.kaggle.com/sohier/wtop-10-iterators)
# 12. [List Comprehensions](https://www.kaggle.com/sohier/wtop-11-list-comprehensions)
# 13. [Generators and Generator Expressions](https://www.kaggle.com/sohier/wtop-12-generators)
# 14. [Modules and Packages](https://www.kaggle.com/sohier/wtop-13-modules-and-packages)
# 15. [Strings and Regular Expressions](https://www.kaggle.com/sohier/wtop-14-strings-and-regex)
# 16. [Preview of Data Science Tools](https://www.kaggle.com/sohier/wtop-15-preview-of-data-science-tools)
# 17. [Resources for Further Learning](https://www.kaggle.com/sohier/wtop-16-further-resources)
# 18. [Appendix: Code To Reproduce Figures](https://www.kaggle.com/sohier/wtop-17-figures)

# ## License
# 
# This material is released under the "No Rights Reserved" [CC0](LICENSE)
# license, and thus you are free to re-use, modify, build-on, and enhance
# this material for any purpose.
# 
# That said, I request (but do not require) that if you use or adapt this material,
# you include a proper attribution and/or citation; for example
# 
# > *A Whirlwind Tour of Python* by Jake VanderPlas (O’Reilly). Copyright 2016 O’Reilly Media, Inc., 978-1-491-96465-1
# 
# Read more about CC0 [here](https://creativecommons.org/share-your-work/public-domain/cc0/).
# 
# *[The redistribution of these notebooks as Kaggle kernels was kindly approved by Jake VanderPlas. Kaggle has made minor changes to clarify details specific to our platform. Any errors are Kaggle's alone.]*
