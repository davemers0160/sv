#ifndef LOAD_SV_DATA_H
#define LOAD_SV_DATA_H

// This loading function assumes that the ground truth image size and the input image sizes do not have to be the same dimensions

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
//#include <thread>
#include <vector>
#include <algorithm>
#include <string>
#include <limits>

// Custom Includes
#include "rgb2gray.h"
//#include "ycrcb_pixel.h"

//#include "add_border.h"
//#include "gaussian_kernel.h"

// dlib includes
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>

extern const uint32_t img_depth;
extern const uint32_t secondary;

void load_sv_data(
    const std::vector<std::vector<std::string>> training_file, 
    const std::string data_directory,
    std::pair<uint32_t, uint32_t> mod_params,
    std::vector<std::array<dlib::matrix<uint8_t>, img_depth>> &t_data,
    std::vector<dlib::matrix<uint16_t>> &gt,
    std::vector<std::pair<std::string, std::string>> &image_files
)
{

    int idx;
    
    //std::string imageLocation;
    std::string left_image_file;
    std::string right_image_file;
    std::string depthmap_file;
    
    // clear out the container for the focus and defocus filenames
    image_files.clear();

    for (idx = 0; idx < training_file.size(); idx++)
    {

        // Read in the in-focus and out-of-focus and the ground truth.  The
        // ground truth is assumed to be the last input in the list this is done 
        // because we may want to add a third or more images instead of just two
        left_image_file = data_directory + training_file[idx][0];
        right_image_file = data_directory + training_file[idx][1];
        depthmap_file = data_directory + training_file[idx][training_file[idx].size()-1];

        image_files.push_back(std::make_pair(left_image_file, right_image_file));
        
        // load in the data files
        std::array<dlib::matrix<uint8_t>, img_depth> t;
        dlib::matrix<uint16_t> tf, td;
        dlib::matrix<dlib::rgb_pixel> left_img, left_tmp, right_img, right_tmp, d2, d2_tmp;
        dlib::matrix<int16_t> horz_gradient, vert_gradient;
        dlib::matrix<uint16_t> g, g_tmp;

        // load the images into an rgb_pixel format
        // Image Watch: @mem((left_tmp.data).data, UINT8, 3, left_tmp.nc(), left_tmp.nr(), left_tmp.nc()*3)
        dlib::load_image(left_tmp, left_image_file);
        dlib::load_image(right_tmp, right_image_file);
        dlib::load_image(g_tmp, depthmap_file);

        // crop the images to the right network size
        long rows = left_tmp.nr();
        long cols = left_tmp.nc(); 

        int row_rem = (rows) % mod_params.first;
        if (row_rem != 0)
        {
            long tr = rows - row_rem + mod_params.second;
            if (tr > rows)
                rows = tr - mod_params.first;
            else
                rows = tr;
        }

        int col_rem = (cols) % mod_params.first;
        if (col_rem != 0)
        {
            long tc = cols - col_rem + mod_params.second;
            if (tc > cols)
                cols = tc - mod_params.first;
            else
                cols = tc;
        }

        left_img.set_size(rows, cols);
        right_img.set_size(rows, cols);
        g.set_size(rows, cols);
        dlib::set_subm(left_img, 0, 0, rows, cols) = dlib::subm(left_tmp, 0, 0, rows, cols);
        dlib::set_subm(right_img, 0, 0, rows, cols) = dlib::subm(right_tmp, 0, 0, rows, cols);
        dlib::set_subm(g, 0, 0, rows, cols) = dlib::subm(g_tmp, 0, 0, rows, cols);
        
        switch (img_depth)
        {
            //case 1:
            //    rgb2gray(left_img, tf);
            //    rgb2gray(right_img, td);
            //    
            //    t[0] = (td+256)-tf;
            //    break;
            
            case 2:
                rgb2gray(left_img, t[0]);
                rgb2gray(right_img, t[1]);
                break;
                
            //case 3:
            //    // get the images size and resize the t array
            //    for (int m = 0; m < img_depth; ++m)
            //    {
            //        t[m].set_size(left_img.nr(), left_img.nc());
            //    }
				
            //    switch(secondary)
            //    {
					//case 0:
     //                   for (long r = 0; r < left_img.nr(); ++r)
     //                   {
     //                       for (long c = 0; c < left_img.nc(); ++c)
     //                       {
     //                           dlib::rgb_pixel p;
     //                           dlib::assign_pixel(p, left_img(r, c));
     //                           dlib::assign_pixel(t[0](r, c), p.red);
     //                           dlib::assign_pixel(t[1](r, c), p.green);
     //                           dlib::assign_pixel(t[2](r, c), p.blue);
     //                       }
     //                   }					
					//	break;
					//	
					//case 1:
     //                   for (long r = 0; r < left_img.nr(); ++r)
     //                   {
     //                       for (long c = 0; c < left_img.nc(); ++c)
     //                       {
     //                           dlib::rgb_pixel p;
     //                           dlib::assign_pixel(p, right_img(r, c));
     //                           dlib::assign_pixel(t[0](r, c), p.red);
     //                           dlib::assign_pixel(t[1](r, c), p.green);
     //                           dlib::assign_pixel(t[2](r, c), p.blue);
     //                       }
     //                   }					
					//	break;
						
      //              case 2:
						//rgb2gray(left_img, t[0]);
						//rgb2gray(right_img, t[1]);
      //                  t[2] = dlib::matrix_cast<uint16_t>(dlib::abs(dlib::matrix_cast<float>(t[1])- dlib::matrix_cast<float>(t[0])));
      //                  break;

      //              case 3:
						//rgb2gray(left_img, t[0]);
						//rgb2gray(right_img, t[1]);
      //                  dlib::sobel_edge_detector(t[0], horz_gradient, vert_gradient);
      //                  dlib::assign_image(t[2], dlib::abs(horz_gradient)+dlib::abs(vert_gradient));
      //                  break;

      //              case 4:
						//rgb2gray(left_img, t[0]);
						//rgb2gray(right_img, t[1]);
      //                  //dlib::spatially_filter_image(t[0], t[2], lap_kernel, 1, true, false);
      //                  break;

                //    default:
                //        break;
                //}
                //break;
                
            case 6:
                // get the images size and resize the t array
                for (int m = 0; m < img_depth; ++m)
                {
                    t[m].set_size(left_img.nr(), left_img.nc());
                }

                switch (secondary)
                {
                    // RGB version with each color channel going into it's own layer
                    case 1:
                        // loop through the images and assign each color channel to one 
                        // of the array's of t
                        for (long r = 0; r < left_img.nr(); ++r)
                        {
                            for (long c = 0; c < left_img.nc(); ++c)
                            {
                                dlib::rgb_pixel p;
                                dlib::assign_pixel(p, left_img(r, c));
                                dlib::assign_pixel(t[0](r, c), p.red);
                                dlib::assign_pixel(t[1](r, c), p.green);
                                dlib::assign_pixel(t[2](r, c), p.blue);
                                dlib::assign_pixel(p, right_img(r, c));
                                dlib::assign_pixel(t[3](r, c), p.red);
                                dlib::assign_pixel(t[4](r, c), p.green);
                                dlib::assign_pixel(t[5](r, c), p.blue);
                            }
                        }
                        break;

                    // YCrCb version with each color channle going into it's own layer
                    //case 2:
                    //    for (long r = 0; r < left_img.nr(); ++r)
                    //    {
                    //        for (long c = 0; c < left_img.nc(); ++c)
                    //        {
                    //            dlib::ycrcb_pixel p;
                    //            dlib::assign_pixel(p, left_img(r, c));
                    //            dlib::assign_pixel(t[0](r, c), p.y);
                    //            dlib::assign_pixel(t[1](r, c), p.cr);
                    //            dlib::assign_pixel(t[2](r, c), p.cb);
                    //            dlib::assign_pixel(p, right_img(r, c));
                    //            dlib::assign_pixel(t[3](r, c), p.y);
                    //            dlib::assign_pixel(t[4](r, c), p.cr);
                    //            dlib::assign_pixel(t[5](r, c), p.cb);
                    //        }
                    //    }
                    //    
                    //    break;

                    // LAB version with each color channle going into it's own layer    
                    //case 3:
                    //    for (long r = 0; r < left_img.nr(); ++r)
                    //    {
                    //        for (long c = 0; c < left_img.nc(); ++c)
                    //        {
                    //            dlib::lab_pixel p;
                    //            dlib::assign_pixel(p, left_img(r, c));
                    //            dlib::assign_pixel(t[0](r, c), p.l);
                    //            dlib::assign_pixel(t[1](r, c), p.a);
                    //            dlib::assign_pixel(t[2](r, c), p.b);
                    //            dlib::assign_pixel(p, right_img(r, c));
                    //            dlib::assign_pixel(t[3](r, c), p.l);
                    //            dlib::assign_pixel(t[4](r, c), p.a);
                    //            dlib::assign_pixel(t[5](r, c), p.b);
                    //        }
                    //    }
                    //    break;

                    default:
                        break;
                }
                break;

            //case 9:
            //    depthmap_file = data_directory + training_file[idx][2];
            //    dlib::load_image(d2_tmp, depthmap_file);
            //    d2.set_size(rows, cols);
            //    dlib::set_subm(d2, 0, 0, rows, cols) = dlib::subm(d2_tmp, 0, 0, rows, cols);

            //    // get the images size and resize the t array
            //    for (int m = 0; m < img_depth; ++m)
            //    {
            //        t[m].set_size(left_img.nr(), left_img.nc());
            //    }

            //    switch (secondary)
            //    {
            //        // RGB version with each color channel going into it's own layer
            //    case 1:
            //        // loop through the images and assign each color channel to one 
            //        // of the array's of t
            //        for (long r = 0; r < left_img.nr(); ++r)
            //        {
            //            for (long c = 0; c < left_img.nc(); ++c)
            //            {
            //                dlib::rgb_pixel p;
            //                dlib::assign_pixel(p, left_img(r, c));
            //                dlib::assign_pixel(t[0](r, c), p.red);
            //                dlib::assign_pixel(t[1](r, c), p.green);
            //                dlib::assign_pixel(t[2](r, c), p.blue);
            //                dlib::assign_pixel(p, right_img(r, c));
            //                dlib::assign_pixel(t[3](r, c), p.red);
            //                dlib::assign_pixel(t[4](r, c), p.green);
            //                dlib::assign_pixel(t[5](r, c), p.blue);
            //                dlib::assign_pixel(p, d2(r, c));
            //                dlib::assign_pixel(t[6](r, c), p.red);
            //                dlib::assign_pixel(t[7](r, c), p.green);
            //                dlib::assign_pixel(t[8](r, c), p.blue);
            //            }
            //        }
            //        break;
            //    }

            //    break;
        }
        
        t_data.push_back(t);
        gt.push_back(g);

    }   // end of the read in data loop

}   // end of load_sv_data

// ----------------------------------------------------------------------------
template<typename T>
void get_gt_min_max(
    std::vector<dlib::matrix<T>>& gt,
    T &min_val,
    T &max_val
)
{
    uint32_t idx;

    T mn, mx;
    min_val = std::numeric_limits<T>::max();
    max_val = std::numeric_limits<T>::min();

    // cycle through each of the groundtruth labels to get the min and max values
    for (idx = 0; idx < gt.size(); ++idx)
    {
        dlib::find_min_and_max(gt[idx], mn, mx);
        
        min_val = (mn < min_val) ? mn : min_val;
        max_val = (mx > max_val) ? mx : max_val;
    }

}   // end of get_gt_min_max

#endif  // LOAD_SV_DATA_H
