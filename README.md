# Retinal Explant Metrics

The following repository is home to the code used for the image analysis of **Improved handling and embedding schemes for cultured murine neuroretinal explants**. The code has been modified for reproducibilty sake, gathering area and length in the same way as in the literature.  

## Requirements

Install from the requirements.txt using:

```pip install -r requirements.txt```

Tested on Python 3.7 and 3.9

## Usage

Run main.py with paths to images and file saves a JSON file of results. Please see `main.py` for additional initial parameters in `main` to alter to your choosing.

```>>> python main.py [img_low_mag_path] [img_high_mag_path] [results.json_path] --addmask [addmask_path] --delmask [delmask_path]```

Results return a list per metric of objects identified top-to-bottom-left-to-right.

Example:

{
    "Area, sq.um": [
        1000.0,
        2000.0
    ],
    "Length, um": [
        50.0,
        100.0
    ],
    "Number of Regions": 2
}

## References

[FilFinder](https://fil-finder.readthedocs.io/en/latest/)

[Scikit Image](https://scikit-image.org/)

## Acknowledgment 

Special thanks to Eliah Shamir, Susan Haller, and Rebecca Marton for their feedback in creating this repo and for their above and beyond work in the production to the paper this repo is based on. Thanks to Genentech Incorporated and it's resources and staff that aided in the development of these methods.

## Contact
Kevin Marroquin (marroquk@gene.com) 
