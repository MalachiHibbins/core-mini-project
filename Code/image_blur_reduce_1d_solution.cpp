#include <time.h>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <mpi.h>

// function to read in image data from a file generated in our jupyter notebook
// input: the name of the file
std::vector < int > read_image(std::string filename, int& Ni, int& Nj){

  //std::cout << "Reading in image data from " << filename << std::endl;

  // open the file
  std::ifstream image_file(filename);

  // read in the size of the image
  image_file >> Ni >> Nj;
  
  // now read in the image data to a single 1D vector
  std::vector < int > image_data;
  std::vector < int > rgb = {0, 0, 0};
  int ii, jj;
  for (int i=0;i<Ni;i++){
    for (int j=0;j<Nj;j++){
      image_file >> ii >> jj >> rgb[0] >> rgb[1] >> rgb[2];
      for (int k=0; k<3; k++){
        image_data.push_back(rgb[k]);
      }
    }
  }
  
  // close the file
  image_file.close();
  
  return image_data;
  
}


// function to output image data in a format which will be read by our jupyter notebook
// inputs: the name of the file and the data
void write_image(std::string filename, std::vector < int > image_data, int Ni, int Nj){

  //std::cout << "Writing image data to " << filename << std::endl;

  // open the file
  std::ofstream image_file(filename);

  // write the size of the image
  image_file << Ni << " "<< Nj << std::endl;
  
  // now write out the image data
  int ind = 0;
  for (int i=0;i<Ni;i++){
    for (int j=0;j<Nj;j++){
      image_file << i << " " << j << " " << image_data[ind] << " " << image_data[ind+1] << " " << image_data[ind+2] << std::endl;
      ind += 3;
    }
  }
  
  // close the file
  image_file.close();

}


// divide the image over MPI tasks using slices
void distribute_grid(int N, int iproc, int nproc, int& i0, int& i1){

  i0 = 0;
  i1 = N;
  if (nproc > 1){
    int ni = N / nproc; 
    i0 = iproc * ni;
    i1 = i0 + ni;
    // make sure we take care of the fact that the grid might not easily divide into slices
    if (iproc == nproc - 1){
      i1 = N;
    }
  }
  
  //std::cout << "MPI task " << iproc << " of " << nproc << " is responsible for " << i0 << "<=i<" << i1 << " (" << i1 - i0 << " rows/columns)" << std::endl;

}

// convert between 3D indices and a 1D index
int get_1D_index(int i, int j, int k, int Nj){
  return i * Nj * 3 + j * 3 + k;
}


void blur_image(int Ni, int Nj, int i0, int i1, int j0, int j1, int iproc, int nproc, std::vector < int > & old_image, std::vector < int >& new_image){

  ////////////////////////////////////////////////////////
  //                 Blur The Image                     //
  ////////////////////////////////////////////////////////  

  for (int i=i0;i<i1;i++){
    for (int j=j0;j<j1;j++){
       
      std::vector < int > neighbours;
      for (int k=0;k<3;k++){
        int ind = get_1D_index(i, j, k, Nj);
        neighbours.push_back(old_image[ind]);
      }         

      if (i > 0){
        for (int k=0;k<3;k++){
          int ind = get_1D_index(i-1, j, k, Nj);
          neighbours.push_back(old_image[ind]);
        }
      }
      if (i < Ni-1){
        for (int k=0;k<3;k++){
          int ind = get_1D_index(i+1, j, k, Nj);
          neighbours.push_back(old_image[ind]);
        }
      }     
      if (j > 0){
        for (int k=0;k<3;k++){
          int ind = get_1D_index(i, j-1, k, Nj);
          neighbours.push_back(old_image[ind]);
        }
      }     
      if (j < Nj-1){
        for (int k=0;k<3;k++){
          int ind = get_1D_index(i, j+1, k, Nj);
          neighbours.push_back(old_image[ind]);
        }
      }  
        
      // we will lose some information by doing integer division, but this is ok as we want integers at the end anyway
      for (int k=0;k<3;k++){
        int average = 0;
        for (int nind=0;nind<neighbours.size();nind+=3){
          average += neighbours[nind+k];      
        }
        average /= (neighbours.size()/3);
        int ind = get_1D_index(i, j, k, Nj);
        new_image[ind] = average;
      }
       
    }
  }
  
  ////////////////////////////////////////////////////////
  //          Communicate The Updated Image             //
  ////////////////////////////////////////////////////////    
    
  if (nproc > 1){
    // perform a single communication
    MPI_Allreduce(new_image.data(), old_image.data(), Ni*Nj*3, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

    // set new_image to zero ready for the next iteration
    for (int ind=0;ind<Ni*Nj*3;ind++){
      new_image[ind] = 0;
    }  

  }
  else{
    // we only have 1 MPI task, so we can copy as before
    old_image = new_image;  
  }
  
}


int main(int argc, char **argv)
{

  ////////////////////////////////////////////////////////
  //                   Initialise MPI                   //
  ////////////////////////////////////////////////////////  

  // initialise MPI
  MPI_Init(&argc, &argv);

  // Get the number of processes in MPI_COMM_WORLD
  int nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // get the rank of this process in MPI_COMM_WORLD
  int iproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
  
  
  ////////////////////////////////////////////////////////
  //                   Read in Image                    //
  ////////////////////////////////////////////////////////   

  // read in the image data on the root task
  int Ni, Nj;
  std::vector < int > old_image;
  if (iproc == 0){
    old_image = read_image("original_image.dat", Ni, Nj);
  }
  
  // broadcast the data
  if (nproc > 1){
    // first share the image size
    MPI_Bcast(&Ni, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nj, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // now broadcast the actual image data
    for (int i=0;i<Ni*Nj*3;i++){
      old_image.push_back(0);
    }
    MPI_Bcast(old_image.data(), Ni*Nj*3, MPI_INT, 0, MPI_COMM_WORLD);      
  }
  
  
  ////////////////////////////////////////////////////////
  //                 Initialise new_image               //
  ////////////////////////////////////////////////////////  

  std::vector < int > new_image(Ni*Nj*3, 0);
  
   
  ////////////////////////////////////////////////////////
  //             Set Number of Iterations               //
  ////////////////////////////////////////////////////////    
  
  // number of iterations
  int N = 10;
  
  
  ////////////////////////////////////////////////////////
  //           Distribute Data Among Tasks              //
  ////////////////////////////////////////////////////////     
  
  // divide the rows among MPI tasks
  int i0, i1;
  distribute_grid(Ni, iproc, nproc, i0, i1);
  int j0 = 0;
  int j1 = Nj;
  
  
  ////////////////////////////////////////////////////////
  //       Run The Model - 1D and 3d Versions           //
  ////////////////////////////////////////////////////////     
    
  // just to be sure the tasks are still in sync
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  
  for (int n=0;n<N;n++){
    blur_image(Ni, Nj, i0, i1, j0, j1, iproc, nproc, old_image, new_image);
  }
  
  double finish = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  
    
  ////////////////////////////////////////////////////////
  //           Write Out The Blurred Image              //
  ////////////////////////////////////////////////////////     
    
  if (iproc == 0){
    // print the time, and the value of 3 pixels as a quick (non-exhaustive) check of correctness
    int ind0 = get_1D_index(Ni/3, Nj/2, 0, Nj);
    int ind1 = get_1D_index(Ni/2, Nj-10, 2, Nj);
    int ind2 = get_1D_index(4, Nj-1, 1, Nj);
    std::cout << nproc << " " << finish - start << " " << old_image[ind0] << " " << old_image[ind1] << " " << old_image[ind2] << std::endl;
    
    // write out the blurred image - note this will in fact be old_image, since this is where the output of the reduction ends up
    write_image("blurred_image.dat", old_image, Ni, Nj);
  }
  
  ////////////////////////////////////////////////////////
  //                    Finalise MPI                    //
  ////////////////////////////////////////////////////////    
          
  // finalise MPI
  MPI_Finalize();
        
  return 0;
}

