# The Virtual 2D Materials Database (V2DB)

> V2DB is a database of two-dimensional materials generated by the approach described by the following paper: 

An artificial intelligence-aided virtual screening recipe for two-dimensional materials discovery
npj Computational Materials – https://doi.org/10.1038/s41524-020-00375-7
(if you are using the codes or data in your research, please cite this paper)

V2DB (latest version) is available at Harvard Dataverse - https://doi.org/10.7910/DVN/SNCZF4

The codes in this repository are developed by Murat Cihan Sorkun and Séverin Astruc.

Training data used for the development of the machine learning (ML) models and selection of prototypes and elements are collected from C2DB (ver. 2018-12-10) [1,2].

# How to Run

All the materails are deposited into CodeOcean for easy reproduction. - https://doi.org/10.24433/CO.7049461.v1

At CodeOcean:
To execute the generation and prediction process simply click the "Reproducible Run" button located at top right corner.
Run button executes the "generation.py", "prediction.py", and the "sqlite.py" scripts respectively.


### Dependencies
- pandas==0.23.0
- scikit-learn==0.19.1
- ase==3.18


# Contents of Repository

There are three steps to generate new materials and predict their properties. They are accomplished by running the following python scripts.
1. **generation/generation.py**  - Generates new 2D materials in CSV files for each prototype.
2. **ML_training/prediction.py**  - Trains and exports the machine learning models for property prediction.
3. **database/sqlite.py**      - Predicts the properties of newly generated materials and stores the results into the SQLite database.

Given three python scripts call functions and classes from the following python files:
- generation/
    - **v2db.py** - Includes functions and classes to generate new 2D materials.
    - **iodb.py** - Includes functions for input and output of generation. 
    - **filtering.py** - Includes functions for filtering (non neutral and duplicate) the new 2D materials.
- ML_training/
    - **formula_to_onehot_vector.py** - Includes functions to convert the materials into a vectors.
    - **metrics.py** - Includes functions which calculate and plot the score of the ML models (regressor or classifier).
    - **preprocessing.py** - Includes functions to retrieve the data from C2DB database which is used for training the ML models.
- database/
    - **formula_to_onehot_vector.py**  - Includes functions to convert the materials into a vectors.
    - **iodb.py** - Includes functions to get the prototype, the charge, and the electronegativity of the new generated materials.
    


## generation/generation.py

This script generates new materials by brute force substitution of selected elements for selected prototypes from the C2DB database. 
Removes the duplicate (according to the symmetry of the structural prototype) and the none neutral materials.

- We grouped each element used as A (positive charged in the materials) or B (negative charged) (see types.csv). 
- We grouped prototypes according to their unitcell-group and their symmetry type (see groups.csv).
- There are 4 groups (AB, ABB, AABB and AABBBB) and 4 symmetry type ( XY, Y, Z and XYY). 

List of unitcell groups, symmetry types and number of prototypes given below:

- ['AB', 'Z']        -> 2 prototypes
- ['ABB', 'Z']       -> 2 prototypes
- ['ABB', 'Y']       -> 3 prototypes
- ['AABB', 'Y']      -> 1 prototypes
- ['AABB', 'XY']     -> 9 prototypes
- ['AABB', 'Z']      -> 1 prototypes  
- ['AABBBB', 'XYY'  ]-> 3 prototypes
- ['AABBBB', 'XY']   -> 1 prototypes


**Important Note:**
It is possible to tune the number of group/symmetry couples to be generated by changing the value inside the brackets `[0:i]` from the file (generation/generation.py). The range of i is from 1 to 8 (generates the number of lines of the list above). Default value is given as 3 in order to reduce to the generation time. By making it 8, it is possible to generate full database. However it takes too much time on CodeOcean. It requires min 8 Gb ram, to reproduce all database in your local. 

Code line:
```python
for group, label in symmetries[0:3]:
    # symmetries[0:3] can be changed to # symmetries[0:8] to generate all materials
```

**inputs**
1. data/types.csv - type (A or B) of each element, from C2DB analyze.
2. data/charges.csv - charges of each element, adapted from [3].
3. data/symmetries.csv - symmetry label for each prototype, from C2DB analyze.
4. data/groups.csv - Group for each prototype, from C2DB analyze.
5. data/c2db.db - C2DB database.

**outputs**
1. results/subpart - the subparts A.csv, AA.csv, B.csv, BB.csv, and the duplicate filtered subparts AAR.csv, BBR.csv, BBBBR.csv.
2. results/generation - CSV for each prototype: the template of the name is group_prototype.csv. For example, in AB_BN.csv prototype is BN which belongs to AB group.


## ML_training/prediction.py

Explanations of predicted properties
- Stability: it is a binary label which says if the material is stable or not. It is used as part of the filtering.
- Gap (eV): Band Gap (PBE)
- VBM (eV): VBM vs vacuum (PBE)
- CBM (eV): CBM vs vacuum (PBE) calculated as bandgap + VBM
- Work function (eV)
- Heat of formation (eV/atom)
- Energy above convex hull (eV/atom)
- Has direct bandgap (Boolean)
- Magnetic State (NM for None Magnetic, FM for ferromagnetic and AFM for antiferromagnetic)
- Is Magnetic (Boolean)


**inputs**
1. data/c2db.db - C2DB database.
2. data/types.csv - type (A or B) of each element, from C2DB analyze.
3. data/en_A.csv and en_B.csv - electronegativity of the A and B elements. [4]

**outputs**
1. results/models - exported models that is used for labeling new materials
2. results/encoders - exported vectors to be used on ml models (to keep the order of elements fixed) 


## database/sqlite.py

This script predicts the properties of the generated materials by importing models trained in previous step, then applies the stability filter by using the results of stability, heat of formation, and energy above convex hull models. Finally, saves the filtered materils into the results/v2db.db database.

**inputs**
1. results/encoders - exported vectors to be used on ml models (to keep the order of elements fixed)
2. results/models - exported models that is used for labeling new materials
3. data/elements.csv - all elements selected from the c2db database.
3. data/elements_nonzero.csv - nonzero bandgap elements selected from the c2db database.
4. data/stoichiometries.csv - stoichiometry for each prototype.
5. data/types.csv  - charge type (A or B) of each element.
6. data/en_A.csv and en_B.csv - electronegativity of the A and B elements [4].

**outputs**
1. results/v2db.db - Generated Virtual 2D Materials Database (V2DB) in sqlite database format



**Note:**
results/v2db.db is a predefined empty database to be filled by new materials.


# License and Copyright
MIT License

Copyright (c) 2019 Murat Cihan Sorkun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# References
[1] Haastrup, S., Strange, M., Pandey, M., Deilmann, T., Schmidt, P. S., Hinsche, N. F., ... & Gath, J. (2018). The Computational 2D Materials Database: high-throughput modeling and discovery of atomically thin crystals. 2D Materials, 5(4), 042002.

[2] https://cmrdb.fysik.dtu.dk/c2db/

[3] Greenwood, Norman N.; Earnshaw, Alan. (1997), Chemistry of the Elements (2nd ed.), Oxford: Butterworth-Heinemann, ISBN 0080379419 p. 28.

[4] Pauling, L. (1932) The nature of the chemical bond. IV. The energy of single bonds and the relative electronegativity of atoms. Journal of the American Chemical Society 54, 3570–3582.

