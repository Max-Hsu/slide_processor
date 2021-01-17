# slide_processor

Hi
this slide_processor will help you from 

**this**
![](https://i.imgur.com/ZgIZJZ6.jpg)


**to this**
![](https://i.imgur.com/l0wUg0u.jpg)

automaticially

## usage
./start.sh ==file choices== (arguments)

### file choices
1. if empty -> process all the file in the directory that user is at
2. some directory -> process all file in that directory
3. certain files -> process the files
4. file filename -> will process the file in the filename


## arguments
if the output wasn't satisfied , change the argument by --mod*
```
  -h, --help            show this help message and exit
  -s, --show            show the picture during processing , any key to
                        proceed
  -o, --output          the template for output image format,must contain %s
                        for seperating files
  -r, --ocr             apply ocr application
  -q, --quiet           don't display anything
  -e, --enhance         enhancing the output image
  --mod_threshold1=THRESHOLD1_BINARY
                        modify first binary threshold value (default :
                        np.mean)
  --mod_threshold2_alpha=THRESHOLD2_ALPHA
                        modify second layer contrast alpha value (default :
                        1.5)
  --mod_threshold2_beta=THRESHOLD2_BETA
                        modify second layer contrast beta value (default : 0)
  --mod_threshold3_laplacian=THRESHOLD3_LAPLACIAN
                        modify third layer laplacian center value (default :
                        24)
  --mod_threshold4_gaussian=THRESHOLD4_GAUSSIAN
                        modify forth layer gaussian blur kernel size (default
                        : 3)
  --mod_threshold5_canny_low=THRESHOLD5_CANNY_LOW
                        modify fifth canny layer low threshold value (default
                        : 10)
  --mod_threshold5_canny_high=THRESHOLD5_CANNY_HIGH
                        modify fifth canny layer high threshold value (default
                        : 15)
  --mod_threshold6_dilate_size=THRESHOLD6_DILATE_SIZE
                        modify sixth layer dilate kernel size : 3)
  --mod_threshold6_dilate_iter=THRESHOLD6_DILATE_ITER
                        modify sixth layer dilate iteration times : 3)
                        ```
