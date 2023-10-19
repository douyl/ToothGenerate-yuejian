This is code for generating facial photograph after orthodontics.

Note:
1. The directory tree is like
      - Code
         - 2DRegist_Test.py (main program)
      - Data
         - Case1
            - teeth       (save the dental model before alignment)
            - teeth_after (save the dental model after alignment)
            - orig.png    (the input photo)
         - Case2
         - ......
         - 2D_coords.txt (save the 牙尖点坐标 of the input facial photo)

2. Ouput photo will be saved in the same folder of Input photo, named *'pred_face.png'*.

3. How to run the code?

   (1) Download the pre-trained model parameters from *https://pan.baidu.com/s/1DuPQgsP3UmD1rN26QO7TwA?pwd=iqyi*. Then put the model parameters under the path *'Code/GenerateTooth/ckpt/ckpt_contour2tooth_v2_ContourSegm_facecolor_lightcolor_10000.pth'*. (Make sure the path and name is correct.) 

   (1) Put the testing case into the folder called 'Data', fomatting as above.

   (2) Change the case name in Line 26 in *'Code/2DRegist_Test.py'*.

   (3) Enter the 牙尖点坐标 into *'Data/2D_coords.txt'*.  **(6 upper teeth is recommended. 4 or 2 upper teeth also works.)**

   (4) Run *'Code/2DRegist_Test.py'*.
