## PA05 - Programming Assignment 5 - Biovision
This project is an optional bonus:
If you complete it, it will replace your lowest grade.
If you do not complete this assginment, it will not hurt your grade.

### Task

#### Program a viable submission to the 2018 Data Science Bowl Competition: "Spot Nuclei. Speed Cures." 

1. Review all the vision-related python links/code I provided, especially the segmentation demos.

2. Peruse the PDFs I provided on biological image processing, starting with the single paper, then as many chapters of the book as you can get through and find rewarding.

3. Learn about the projcet.
    * Read all about the competition by systematically perusing all the tabs on this site:
    https://datasciencebowl.com/
    * Enjoy a little hype:
    https://www.youtube.com/watch?time_continue=7&v=eHwkfhmJexs
    * Watch this more informative video:
    https://www.youtube.com/watch?time_continue=93&v=Dbiq6l50zO8
    * Read all the sub-sections of the overview tab on this page:
    https://www.kaggle.com/c/data-science-bowl-2018

4. Download and get comfortable with the data. 
What are the data formats?
How is the mask (label) of the train answers encoded?
Don't proceed until you have that figured out!
Check out:
    * `stage1_sample_submission.csv`
    * https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    * https://en.wikipedia.org/wiki/Run-length_encoding

5. Review the "Kernels" tab of the below link, as well as the "Discussion" tab for information about other participants' experiences and thoughts on the data and its analysis. 
The competition ended, so we have missed the window to win the $100,000 prize, but that is actually beneficial for being able to see the set of solutions that did well and those that did not. 
You should get a basic submission using primitive and skimage methods from class working, then if you want, you can copy/explore more advanced methods from other "kernels" in kaggle.
You can take functions like RLE from other notebooks, but try to do the basics from scratch yourself, before exploring other more advanced methods.
    * You can try to check out a basic notebook like: https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies
    * https://www.kaggle.com/c/data-science-bowl-2018
    * https://datasciencebowl.com/2018tutorialwinners/

6. Using your choice of Python environment and libraries, attempt to satisfy the goals of the competition. 
You can use pre-built libraries, or do most from scratch, whichever approach you are interested in.
I suggest trying a primitive histogram based method first.

7. I am "giving" you the full autograder on this one (just not the test labels); 
... the autograder is actually kaggle's, but it's the same as I would do :)
Check your performance on Kaggle using their submission checker, where you may submit a maximum of 5 entries per day.
Kaggle wants you to subimt the test2 data answers. 
    * https://www.kaggle.com/c/data-science-bowl-2018/submit 

8. Submit any related files, `*.py`, `*.ipynb`, `*.csv`, etc. to this repository.
Things you are required to submit include:
    * `pa05-cell-vision.py` or `pa05-cell-vision.ipynb` and any supporting python files you want to include in the repo, `git add *.py`
    * `my_working_kaggle_submission.csv` submission file, with format, etc defined in the competition rules.
    * `my_working_kaggle_submission_result.png` a Screenshot of your displayed performance on the Kaggle site, as judged by the automated submission system's grading of your submitted CSV (the one from the previous bullet).
    * `pa05-report.md` a PDF report detailing your approach, methods, results, interpretation, and any difficulties encountered. Please include all project members names at the top.
    * Please **DO NOT** submit the massive data files! Do not `$ git add .` please.

**Have fun!**

### Grading
* Grading will be based on a demonstration of significant quantity of effort and exploration.
* For this project, you will certainly get a good grade for doing objectively well (given you use your own code, or have built appreciably on another solution).
The competition rules suggest maximizing your https://en.wikipedia.org/wiki/Jaccard_index is the way to go.
* However, if you demonstrate significant (and reasonably correct) effort into an approach that is experimental, new to you, or generally novel, but don't have great performance, then you can still do very well.  
* Try to get the basic inputs, outputs, and formats in place, so you can actually evaluate your own performance.

### Execution
* We'll run this with python3.
