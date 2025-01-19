MODULE def_kind
!
! Definition:
!
!   This is to guarantee portability on different machines. P indicates "precision"
!
! Dependencies:
!
!   All subroutines and modules
!
! Author: W. Imperatori
!
! Modified: W.Imperatori, January 2009 (v1.3)

implicit none

! Real single precision
integer,parameter:: r_single=selected_real_kind(p=6)
! Real double precision
integer,parameter:: r_double=selected_real_kind(p=15)
! Real 'scattering' precision (i.e. only 4 digits after decimal point)
!integer,parameter:: r_scat=selected_real_kind(p=4)    
! Integer single representation (i.e. up to 10**9 order)
integer,parameter:: i_single=selected_int_kind(9)  
! Integer double representation (i.e. up to 10**18 order)
integer,parameter:: i_double=selected_int_kind(18)

END MODULE def_kind

!===================================================================================================

MODULE fast_marching
!
!   Contains global variables for the fast marching method  
!
! Dependencies:
!
!   Module def_kind
!
! Author: W.Imperatori, October 2014 (v1.6)
!
! Modified: 
!

use def_kind

implicit none

save

! smaller and larger representable numbers
real(kind=r_single),parameter                      :: scalar = 1.0
real(kind=r_single),parameter                      :: eps = tiny(scalar), INF = huge(scalar)
! array defining the "narrow-band" 
integer(kind=i_single),allocatable,dimension(:,:,:):: Frozen
! flags for fast marching accuracy 
logical                                            :: usesecond,usecross
! array storing the shape of input velocity model
integer(kind=i_single),dimension(3)                :: dims  

! array storing arrival times
real(kind=r_single),allocatable,dimension(:,:,:)   :: T

END MODULE fast_marching

!=========================================================================================

module type_a_def 

   use def_kind

   implicit none 
   
   type type_a 
      real(kind=r_single) :: data
      integer,dimension(3):: index
      type(type_a),pointer:: left => null() 
      type(type_a),pointer:: right => null() 
   end type type_a 

end module type_a_def 

module type_a_ops 

   use def_kind; use type_a_def 

   implicit none 

   interface operator(<=) 
      module procedure less_than_or_equals 
   end interface operator(<=) 

   interface operator(==)
      module procedure equal_to
   end interface operator(==)

   interface assignment(=) 
      module procedure assign 
   end interface assignment(=) 

   contains 

      function less_than_or_equals(op1,op2) 
         logical less_than_or_equals 
         type(type_a),intent(in):: op1 
         type(type_a),intent(in):: op2 
         less_than_or_equals = op1%data <= op2%data 
      end function less_than_or_equals

      function equal_to(op1,op2)
         logical equal_to
         type(type_a),intent(in):: op1
         type(type_a),intent(in):: op2
         equal_to = ((op1%data <= op2%data) .and. all(op1%index .ne. op2%index))
      end function equal_to

      subroutine assign(op1, op2) 
         type(type_a),intent(out):: op1 
         type(type_a),intent(in) :: op2 
         op1%data = op2%data 
         op1%left => op2%left 
         op1%right => op2%right 
      end subroutine assign 
      
end module type_a_ops
 
module tree_funcs 

   use type_a_ops 

   implicit none 

   contains 

      recursive subroutine add_node(root, node) 

         type(type_a),pointer:: root 
         type(type_a),target :: node 

         if(.NOT.associated(root)) then 
            root => node 
         else if(node <= root) then 
            call add_node(root%left, node) 
         else 
            call add_node(root%right, node) 
         end if 

      end subroutine add_node 

      recursive subroutine delete_node(root, node) 

         type(type_a),pointer   :: root 
         type(type_a),intent(in):: node 
         type(type_a),pointer   :: temp 

         if(.NOT.associated(root)) then 
!             print*,'error: node not found!'
!             stop
!            write(*,'(a)') ' Node not found' 
            return 
         end if

         if(node <= root) then
            if( (root <= node) .and. (all(root%index .eq. node%index)) ) then
               if(all(root%index .eq. node%index)) then               
                  if(.NOT.associated(root%left)) then 
                     temp => root 
                     root => root%right 
                     nullify(temp%right) 
                     deallocate(temp) 
                  else if(.NOT.associated(root%right)) then 
                     temp => root 
                     root => root%left 
                     nullify(temp%left) 
                     deallocate(temp) 
                  else 
                     temp => delete_and_return_biggest(root%left) 
                     temp%left => root%left 
                     temp%right => root%right 
                     root%left => temp 
                     temp => root 
                     root => root%left 
                     nullify(temp%left) 
                     nullify(temp%right) 
                     deallocate(temp) 
                  endif
                  return
               endif
            else 
               call delete_node(root%left, node) 
            endif 
         else 
            call delete_node(root%right, node) 
         endif 
         
      end subroutine delete_node 
      
      recursive function delete_and_return_biggest(root) result(temp) 

         type(type_a),pointer:: temp 
         type(type_a),pointer:: root 

         if(.NOT.associated(root%right)) then 
            temp => root 
            root => root%left 
            nullify(temp%left) 
         else 
            temp => delete_and_return_biggest(root%right) 
         end if 

      end function delete_and_return_biggest 

      recursive subroutine print_tree(root) 

         type(type_a),pointer :: root

         if(.NOT.associated(root)) then 
            write(*,'(a)') ' Empty tree' 
         else 
            if(associated(root%left)) then
               call print_tree(root%left)
            endif
            write(*,*) root%data,root%index
            if(associated(root%right)) then 
               call print_tree(root%right)
            endif
         endif 

      end subroutine print_tree

      recursive subroutine get_minimum(root,index,val)

         type(type_a),pointer                           :: root
         real(kind=r_single),intent(out)                :: val
         integer(kind=i_single),dimension(3),intent(out):: index

         if(associated(root%left)) then
            call get_minimum(root%left,index,val)
         else
            val = root%data; index = root%index
         endif

      end subroutine get_minimum 
 
end module tree_funcs

!=========================================================================================

PROGRAM ffm_stdalone
!-----------------------------------------------------------------------------------------
! Program to compute first arrival times using the Fast Marching Method (FMM). The 
! implementation follows the paper "Multistencils Fast Marching Methods: a highly accurate
! solution to the Eikonal equation on Cartesian coordinates", Sabry-Hassouna et Farag, IEEE,
! 2007.
! 
! Possible configurations are: 1st-order stencils (standard FMM)
!                              2nd-order stencils (high-accuracy FMM - FMMHA)
!                              1st-order multi-stencils (MSFMM)
!                              2nd-order multi-stencils (MSFMMHA)
!
! Default configuration is 2nd-order stencils (FMMHA). According to my own tests, MSFMM and 
! MSFMMHA return similar or even less accurate results.
!
! Above configurations are controlled via hard-wired boolean variable "usesecond" and 
! "usecross".
!
! Required input parameters are: Velocity model (F), units: m/s or km/s
!                                Source(s) position (SourcePoints), units: m or km
!                                Number of sources (n_sources)
!                                Grid-step of velocity model (dh), units: m or km
!
! Output consists in 3D array containing first arrival times (T).
!
! The code can handle 2D models as well, as long as they are passed as 3D arrays whose shape
! is n x m x 1. Output array T will have identical shape.
!
! The method is unconditionally stable and allows for multiple sources.  
!
! Version 1.0: W. Imperatori, ETHZ, September 2014
!
!-----------------------------------------------------------------------------------------

!use omp_lib

use def_kind; use fast_marching; use tree_funcs

implicit none

! array with velocity values
real(kind=r_single),allocatable,dimension(:,:,:):: F

! source position
integer(kind=i_single),allocatable,dimension(:,:):: SourcePoints

! grid-step for velocity model
real(kind=r_single)                             :: dh

real(kind=r_single)                             :: Tt

integer(kind=i_single)                          :: n_sources
integer(kind=i_single),dimension(3)             :: NB_min_xyz
                    
real(kind=r_single)                             :: val

! stencils for neighboring points in 3D
integer(kind=i_single),dimension(18)            :: ne
integer(kind=i_single),dimension(36)            :: nw
integer(kind=i_single),dimension(24)            :: nv
  
integer(kind=i_single)                          :: w,p,x,y,z,i,j,k 

character(len=99)                               :: input_file,output_file,vel_file

integer(kind=i_single)                          :: scratch

logical                                         :: isntfrozen3d

real(kind=r_double)                             :: t_s,t_e

type(type_a),pointer                            :: container => NULL() 
type(type_a),pointer                            :: obj => NULL()

!--------------------------------------------------------------------

! initialize array some arrays
ne = (/-1,0,0,1,0,0,0,-1,0,0,1,0,0,0,-1,0,0,1/)
nw = (/-1,0,1,0,-1,1,1,-1,-1,0,1,0,0,-1,0,1,-1,-1,1,1,0,-1,0,1,-1,-1,-1,-1,0,0,0,0,1,1,1,1/) 
nv = (/-1,1,1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1/)
      
usesecond = .TRUE.; usecross = .FALSE.

open(1,file='input.inp',status='unknown')
read(1,*) vel_file                       ! input velocity file
read(1,*) dh                             ! grid-step in velocity model
read(1,*) dims(1)                        ! points along X, Y and Z
read(1,*) dims(2)                        !
read(1,*) dims(3)                        !
read(1,*) n_sources                      ! number of point-sources 

allocate(SourcePoints(3,n_sources))

do i = 1,n_sources
   read(1,*) SourcePoints(1,i)
   read(1,*) SourcePoints(2,i)
   read(1,*) SourcePoints(3,i)
enddo
close(1)   

print*,'Summary of input parameters'
print*,'Velocity file: ',trim(vel_file)
print*,'Grid-step: ',dh
print*,'Point along X-Y-Z in velocity file: ',dims
print*,'Number of sources: ',n_sources

! allocate array for speed values
allocate(F(dims(1),dims(2),dims(3)))

! detect record length
inquire(iolength=scratch) (/real(1)/); scratch = scratch * dims(1)

! open binary file
open(1,file=trim(vel_file),status='old',access='direct',form='unformatted',recl=scratch)

do k = 1,dims(3)
   do j = 1,dims(2)
      read(1,rec = (k-1) * dims(2) + j) F(:,j,k)
   enddo
enddo                  

close(1)  !close file

print*,'Minimum and maximum values in velocity model',minval(F),maxval(F)

! allocate output array
allocate(T(dims(1),dims(2),dims(3)))

! allocate array "frozen"
allocate(Frozen(dims(1),dims(2),dims(3)))
 
forall(i=1:dims(1), j=1:dims(2), k=1:dims(3))   
   Frozen(i,j,k) = 0; T(i,j,k) = INF    
end forall
   
call cpu_time(t_s)   
   
! define initial narrow band   
do p = 1,n_sources                                                       

   ! find source position in grid-points 
   !x = nint(SourcePoints(1,p) / dh) + 1
   !y = nint(SourcePoints(2,p) / dh) + 1
   !z = nint(SourcePoints(3,p) / dh) + 1
   
   x = SourcePoints(1,p); y = SourcePoints(2,p); z = SourcePoints(3,p); 

   ! Set values for source
   Frozen(x,y,z) = 2; T(x,y,z) = 0.

   ! initialise 6 neighbours around the point-source (cube faces)
   do w = 1,6

      i = x + ne(w); j = y + ne(w+6); k = z + ne(w+12)    
      
      ! check if current neighbor is not yet frozen AND inside the domain
      if(isntfrozen3d(i,j,k)) then
   
         ! assign arrival
         T(i,j,k) = 1.0 / F(i,j,k)  
                 
         ! now it is part of the NB                                
         Frozen(i,j,k) = 1  
         
         allocate(obj); obj%data = T(i,j,k); obj%index = (/i,j,k/)

         ! insert element in the list at the right position (ascending order)
         call add_node(container,obj)

      endif
         
   enddo 
   
   ! initialise 12 neighbours around the point-source (cube edges)
   do w = 1,12
   
      i = x + nw(w); j = y + nw(w+12); k = z + nw(w+24)
      
      if(isntfrozen3d(i,j,k)) then
      
         T(i,j,k) = sqrt(2.0) / F(i,j,k); Frozen(i,j,k) = 1
         allocate(obj); obj%data = T(i,j,k); obj%index = (/i,j,k/); call add_node(container,obj)
         
      endif
   enddo      

   ! initialise 8 neighbours around the point-source (cube vertex)
   do w = 1,8
      
      i = x + nv(w); j = y + nv(w+8); k = z + nv(w+16)
      
      if(isntfrozen3d(i,j,k)) then
      
         T(i,j,k) = sqrt(3.0) / F(i,j,k); Frozen(i,j,k) = 1
         allocate(obj); obj%data = T(i,j,k); obj%index = (/i,j,k/); call add_node(container,obj)
         
      endif
   enddo

enddo      
                
! apply FMM as long as arrival-times for all grid-point are computed
do p = 1,dims(1)*dims(2)*dims(3)-1  
     
   ! get first element in the list
   call get_minimum(container,NB_min_xyz,val)
  
   x = NB_min_xyz(1); y = NB_min_xyz(2); z = NB_min_xyz(3)
   
   allocate(obj); obj%index = (/x,y,z/); obj%data = val; call delete_node(container,obj) 

   ! the new point is marked as Frozen
   Frozen(x,y,z) = 2

   ! loop over all 6 neighbors of newly frozen grid-point
   do w = 1,6
      
      !location of neighbor
      i = x + ne(w); j = y + ne(w+6); k = z + ne(w+12)        

      ! Check if current neighbor is not yet frozen and inside the grid 
      if(isntfrozen3d(i,j,k)) then
         
         ! grid-point is now part of NB (if not already before)
         Frozen(i,j,k) = 1
         
         ! compute travel-time    
         call calculate_distance(F(i,j,k),i,j,k,Tt) 
         
         ! update list only if new value for the current node is smaller than previous one
         ! for the same node
         if(Tt .lt. T(i,j,k)) then
            allocate(obj); obj%data = T(i,j,k); obj%index = (/i,j,k/)
            call delete_node(container,obj) 
            obj%data = Tt
            call add_node(container,obj)
            T(i,j,k) = Tt
         endif
         
      endif
      
   enddo                  
                   
enddo ! end main loop                       
                         
call cpu_time(t_e); print*,'First arrivals computed in (sec): ',(t_e - t_s)                          
                         
deallocate(Frozen); deallocate(F)

! scale to actual distance 
T = T * dh

! output result
open(1,file='first_arrival.out',status='unknown',access='direct',form='unformatted',recl=scratch)

do k = 1,dims(3)
   do j = 1,dims(2)
      write(1,rec = (k - 1) * dims(2) + j) T(:,j,k) 
   enddo
enddo   
close(1)  !close file

print*,'Minimum and maximum travel-time',minval(T),maxval(T)

END PROGRAM ffm_stdalone

!=========================================================================================

FUNCTION second_derivative(Txm1,Txm2,Txp1,Txp2)

use def_kind; use fast_marching, only: INF

implicit none

real(kind=r_single),intent(in) :: Txm1,Txm2,Txp1,Txp2
real(kind=r_single)            :: second_derivative
logical                        :: ch1,ch2

!----------------------------------------------------------------------

second_derivative = INF
ch1 = ((Txm2 .lt. Txm1) .and. (Txm1 .lt. second_derivative))
ch2 = ((Txp2 .lt. Txp1) .and. (Txp1 .lt. second_derivative))

if( (ch1 .eqv. .TRUE.) .and. (ch1 .eqv. .TRUE.) ) then
   second_derivative = min( (4.0 * Txm1 - Txm2) / 3.0, (4.0 * Txp1 - Txp2) / 3.0 )
elseif (ch1 .eqv. .TRUE.) then
   second_derivative = (4.0 * Txm1 - Txm2) / 3.0
elseif (ch2 .eqv. .TRUE.) then
   second_derivative = (4.0 * Txp1 - Txp2) / 3.0
endif
      
END FUNCTION second_derivative

!=========================================================================================

SUBROUTINE calculate_distance(Fijk,i,j,k,Tt)

use def_kind; use fast_marching, only: T,usesecond,usecross,INF,eps

implicit none

integer(kind=i_single)              :: q,p,in,jn,kn

real(kind=r_single),dimension(18)   :: Tm = (/0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0./) 
real(kind=r_single),dimension(18)   :: Tm2 = (/0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0./)

real(kind=r_single),dimension(3)    :: Coeff

! local derivatives
real(kind=r_single)                 :: Txm1,Txm2,Txp1,Txp2,Tym1,Tym2,Typ1,Typ2,Tzm1,Tzm2,Tzp1,Tzp2

! local cross-derivatives
real(kind=r_single)                 :: Tr2t1m1, Tr2t1m2, Tr2t1p1, Tr2t1p2, Tr2t2m1, Tr2t2m2, Tr2t2p1, Tr2t2p2
real(kind=r_single)                 :: Tr2t3m1, Tr2t3m2, Tr2t3p1, Tr2t3p2, Tr3t1m1, Tr3t1m2, Tr3t1p1, Tr3t1p2
real(kind=r_single)                 :: Tr3t2m1, Tr3t2m2, Tr3t2p1, Tr3t2p2, Tr3t3m1, Tr3t3m2, Tr3t3p1, Tr3t3p2
real(kind=r_single)                 :: Tr4t1m1, Tr4t1m2, Tr4t1p1, Tr4t1p2, Tr4t2m1, Tr4t2m2, Tr4t2p1, Tr4t2p2
real(kind=r_single)                 :: Tr4t3m1, Tr4t3m2, Tr4t3p1, Tr4t3p2, Tr5t1m1, Tr5t1m2, Tr5t1p1, Tr5t1p2
real(kind=r_single)                 :: Tr5t2m1, Tr5t2m2, Tr5t2p1, Tr5t2p2, Tr5t3m1, Tr5t3m2, Tr5t3p1, Tr5t3p2
real(kind=r_single)                 :: Tr6t1m1, Tr6t1m2, Tr6t1p1, Tr6t1p2, Tr6t2m1, Tr6t2m2, Tr6t2p1, Tr6t2p2
real(kind=r_single)                 :: Tr6t3m1, Tr6t3m2, Tr6t3p1, Tr6t3p2

real(kind=r_single),intent(out)     :: Tt
real(kind=r_single)                 :: Tt2

! return root of polynomial
real(kind=r_single),dimension(2)    :: ansroot = (/0.,0./)

integer(kind=i_single),dimension(18):: Order = (/0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0/)
integer(kind=i_single),dimension(18):: ne = (/-1,0,0,1,0,0,0,-1,0,0,1,0,0,0,-1,0,0,1/)

! Stencil constants
real(kind=r_single),dimension(18)   :: G1 = (/1.,1.,1.,1.,0.5,0.5,1.,0.5,0.5,1.,0.5,0.5,0.5,0.3333333333333, &
                                          &   0.3333333333333,0.5,0.3333333333333,0.3333333333333/)
real(kind=r_single),dimension(18)   :: G2 = (/2.250,2.250,2.250,2.250,1.125,1.125,2.250,1.125,1.125,2.250, &
                                          &   1.125,1.125,1.125,0.750,0.750,1.125,0.750,0.750/)

real(kind=r_single),intent(in)      :: Fijk
integer(kind=i_single),intent(in)   :: i,j,k

integer(kind=i_single)              :: minarray

logical                             :: isfrozen3d,IsFinite,IsInf

real(kind=r_single)                 :: second_derivative

!-----------------------------------------------------------------------------------------

! get first order derivatives (use only frozen points)
in = i-1; jn = j+0; kn = k+0
if(isfrozen3d(in,jn,kn)) then
   Txm1 = T(in,jn,kn) 
else
   Txm1 = INF
endif   
                      
in = i+1; jn = j+0; kn = k+0 
if(isfrozen3d(in,jn,kn)) then
   Txp1 = T(in,jn,kn)
else
   Txp1 = INF
endif

in = i+0; jn = j-1; kn = k+0
if(isfrozen3d(in,jn,kn)) then
   Tym1 = T(in,jn,kn)
else
   Tym1 = INF
endif

in = i+0; jn = j+1; kn = k+0
if(isfrozen3d(in,jn,kn)) then
   Typ1 = T(in,jn,kn)
else 
   Typ1 = INF
endif   

in = i+0; jn = j+0; kn = k-1
if(isfrozen3d(in,jn,kn)) then
   Tzm1 = T(in,jn,kn)
else
   Tzm1 = INF
endif   

in = i+0; jn = j+0; kn = k+1
if(isfrozen3d(in,jn,kn)) then
   Tzp1 = T(in,jn,kn)
else
   Tzp1 = INF
endif

if(usecross) then
    Tr2t1m1 = Txm1; Tr2t1p1 = Txp1
    in = i-0; jn = j-1; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr2t2m1 = T(in,jn,kn); else; Tr2t2m1 = INF; endif

    in = i+0; jn = j+1; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr2t2p1 = T(in,jn,kn); else; Tr2t2p1 = INF; endif
       
    in = i-0; jn = j-1; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr2t3m1 = T(in,jn,kn); else; Tr2t3m1 = INF; endif
   
    in = i+0; jn = j+1; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr2t3p1 = T(in,jn,kn); else; Tr2t3p1 = INF; endif

    Tr3t1m1 = Tym1; Tr3t1p1 = Typ1
    in = i-1; jn = j+0; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr3t2m1 = T(in,jn,kn); else; Tr3t2m1 = INF; endif

    in = i+1; jn = j+0; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr3t2p1 = T(in,jn,kn); else; Tr3t2p1 = INF; endif
   
    in = i-1; jn = j-0; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr3t3m1 = T(in,jn,kn); else; Tr3t3m1 = INF; endif

    in = i+1; jn = j+0; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr3t3p1 = T(in,jn,kn); else; Tr3t3p1 = INF; endif

    Tr4t1m1 = Tzm1; Tr4t1p1 = Tzp1
    in = i-1; jn = j-1; kn = k-0
    if(isfrozen3d(in,jn,kn)) then; Tr4t2m1 = T(in,jn,kn); else; Tr4t2m1 = INF; endif

    in = i+1; jn = j+1; kn = k+0
    if(isfrozen3d(in,jn,kn)) then; Tr4t2p1 = T(in,jn,kn); else; Tr4t2p1 = INF; endif

    in = i-1; jn = j+1; kn = k-0
    if(isfrozen3d(in,jn,kn)) then; Tr4t3m1 = T(in,jn,kn); else; Tr4t3m1 = INF; endif

    in = i+1; jn = j-1; kn = k+0
    if(isfrozen3d(in,jn,kn)) then; Tr4t3p1 = T(in,jn,kn); else; Tr4t3p1 = INF; endif
     
    Tr5t1m1 = Tr3t3m1; Tr5t1p1 = Tr3t3p1
    in = i-1; jn = j-1; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr5t2m1 = T(in,jn,kn); else; Tr5t2m1 = INF; endif
   
    in = i+1; jn = j+1; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr5t2p1 = T(in,jn,kn); else; Tr5t2p1 = INF; endif
   
    in = i-1; jn = j+1; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr5t3m1 = T(in,jn,kn); else; Tr5t3m1 = INF; endif
  
    in = i+1; jn = j-1; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr5t3p1 = T(in,jn,kn); else; Tr5t3p1 = INF; endif

    Tr6t1m1 = Tr3t2p1; Tr6t1p1 = Tr3t2m1
    in = i-1; jn = j-1; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr6t2m1 = T(in,jn,kn); else; Tr6t2m1 = INF; endif
       
    in = i+1; jn = j+1; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr6t2p1 = T(in,jn,kn); else; Tr6t2p1 = INF; endif
       
    in = i-1; jn = j+1; kn = k-1
    if(isfrozen3d(in,jn,kn)) then; Tr6t3m1 = T(in,jn,kn); else; Tr6t3m1 = INF; endif
   
    in = i+1; jn = j-1; kn = k+1
    if(isfrozen3d(in,jn,kn)) then; Tr6t3p1 = T(in,jn,kn); else; Tr6t3p1 = INF; endif
    
endif 
    
!The values in order is 0 if no neighbors in that direction, 1 if 1st order derivatives are used and 
! 2 if 2nd order derivatives are used 
    
! Mark 1st order derivatives in x and y direction 
Tm(1) = min(Txm1,Txp1); if(IsFinite(Tm(1))) then; Order(1) = 1; else; Order(1) = 0; endif
Tm(2) = min(Tym1,Typ1); if(IsFinite(Tm(2))) then; Order(2) = 1; else; Order(2) = 0; endif
Tm(3) = min(Tzm1,Tzp1); if(IsFinite(Tm(3))) then; Order(3) = 1; else; Order(3) = 0; endif  
    
! Mark 1st order derivatives in cross directions 
if(usecross) then
   Tm(4) = Tm(1); Order(4) = Order(1)
   Tm(5) = min(Tr2t2m1,Tr2t2p1); if(IsFinite(Tm(5))) then; Order(5) = 1; else; Order(5) = 0; endif
   Tm(6) = min(Tr2t3m1,Tr2t3p1); if(IsFinite(Tm(6))) then; Order(6) = 1; else; Order(6) = 0; endif
        
   Tm(7) = Tm(2); Order(7) = Order(2)
   Tm(8) = min(Tr3t2m1,Tr3t2p1); if(IsFinite(Tm(8))) then; Order(8) = 1; else; Order(8) = 0; endif
   Tm(9) = min(Tr3t3m1,Tr3t3p1); if(IsFinite(Tm(9))) then; Order(9) = 1; else; Order(9) = 0; endif
        
   Tm(10) = Tm(3); Order(10) = Order(3)
   Tm(11) = min(Tr4t2m1,Tr4t2p1); if(IsFinite(Tm(11))) then; Order(11) = 1; else; Order(11) = 0; endif
   Tm(12) = min(Tr4t3m1,Tr4t3p1); if(IsFinite(Tm(12))) then; Order(12) = 1; else; Order(12) = 0; endif
        
   Tm(13) = Tm(9); Order(13) = Order(9)
   Tm(14) = min(Tr5t2m1,Tr5t2p1); if(IsFinite(Tm(14))) then; Order(14) = 1; else; Order(14) = 0; endif
   Tm(15) = min(Tr5t3m1,Tr5t3p1); if(IsFinite(Tm(15))) then; Order(15) = 1; else; Order(15) = 0; endif
        
   Tm(16) = Tm(8); Order(16) = Order(8)
   Tm(17) = min(Tr6t2m1,Tr6t2p1); if(IsFinite(Tm(17))) then; Order(17) = 1; else; Order(17) = 0; endif
   Tm(18) = min(Tr6t3m1,Tr6t3p1); if(IsFinite(Tm(18))) then; Order(18) = 1; else; Order(18) = 0; endif
endif   
    
! Compute 2nd order derivatives 
if(usesecond) then

   ! Get Second order derivatives (only use frozen pixel) 
   in = i-2; jn = j+0; kn = k+0; 
   if(isfrozen3d(in,jn,kn)) then; Txm2 = T(in,jn,kn); else; Txm2 = INF; endif

   in = i+2; jn = j+0; kn = k+0
   if(isfrozen3d(in,jn,kn)) then; Txp2 = T(in,jn,kn); else; Txp2 = INF; endif
   
   in = i+0; jn = j-2; kn = k+0
   if(isfrozen3d(in,jn,kn)) then; Tym2 = T(in,jn,kn); else; Tym2 = INF; endif
   
   in = i+0; jn = j+2; kn = k+0
   if(isfrozen3d(in,jn,kn)) then; Typ2 = T(in,jn,kn); else; Typ2 = INF; endif

   in = i+0; jn = j+0; kn = k-2
   if(isfrozen3d(in,jn,kn)) then; Tzm2 = T(in,jn,kn); else; Tzm2 = INF; endif

   in = i+0; jn = j+0; kn = k+2
   if(isfrozen3d(in,jn,kn)) then; Tzp2 = T(in,jn,kn); else; Tzp2 = INF; endif
        
   if(usecross) then
      Tr2t1m2 = Txm2; Tr2t1p2 = Txp2
      in = i-0; jn = j-2; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr2t2m2 = T(in,jn,kn); else; Tr2t2m2 = INF; endif
      
      in = i+0; jn = j+2; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr2t2p2 = T(in,jn,kn); else; Tr2t2p2 = INF; endif

      in = i-0; jn = j-2; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr2t3m2 = T(in,jn,kn); else; Tr2t3m2 = INF; endif

      in = i+0; jn = j+2; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr2t3p2 = T(in,jn,kn); else; Tr2t3p2 = INF; endif
  
      Tr3t1m2 = Tym2; Tr3t1p2 = Typ2
      in = i-2; jn = j+0; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr3t2m2 = T(in,jn,kn); else; Tr3t2m2 = INF; endif

      in = i+2; jn = j+0; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr3t2p2 = T(in,jn,kn); else; Tr3t2p2 = INF; endif
  
      in = i-2; jn = j-0; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr3t3m2 = T(in,jn,kn); else; Tr3t3m2 = INF; endif
      
      in = i+2; jn = j+0; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr3t3p2 = T(in,jn,kn); else; Tr3t3p2 = INF; endif

      Tr4t1m2 = Tzm2; Tr4t1p2 = Tzp2
      in = i-2; jn = j-2; kn = k-0
      if(isfrozen3d(in,jn,kn)) then; Tr4t2m2 = T(in,jn,kn); else; Tr4t2m2 = INF; endif
      
      in = i+2; jn = j+2; kn = k+0
      if(isfrozen3d(in,jn,kn)) then; Tr4t2p2 = T(in,jn,kn); else; Tr4t2p2 = INF; endif

      in = i-2; jn = j+2; kn = k-0
      if(isfrozen3d(in,jn,kn)) then; Tr4t3m2 = T(in,jn,kn); else; Tr4t3m2 = INF; endif

      in = i+2; jn = j-2; kn = k+0
      if(isfrozen3d(in,jn,kn)) then; Tr4t3p2 = T(in,jn,kn); else; Tr4t3p2 = INF; endif

      Tr5t1m2 = Tr3t3m2; Tr5t1p2 = Tr3t3p2
      in = i-2; jn = j-2; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr5t2m2 = T(in,jn,kn); else; Tr5t2m2 = INF; endif

      in = i+2; jn = j+2; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr5t2p2 = T(in,jn,kn); else; Tr5t2p2 = INF; endif

      in = i-2; jn = j+2; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr5t3m2 = T(in,jn,kn); else; Tr5t3m2 = INF; endif

      in = i+2; jn = j-2; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr5t3p2 = T(in,jn,kn); else; Tr5t3p2 = INF; endif

      Tr6t1m2 = Tr3t2p2; Tr6t1p2 = Tr3t2m2
      in = i-2; jn = j-2; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr6t2m2 = T(in,jn,kn); else; Tr6t2m2 = INF; endif

      in = i+2; jn = j+2; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr6t2p2 = T(in,jn,kn); else; Tr6t2p2 = INF; endif

      in = i-2; jn = j+2; kn = k-2
      if(isfrozen3d(in,jn,kn)) then; Tr6t3m2 = T(in,jn,kn); else; Tr6t3m2 = INF; endif

      in = i+2; jn = j-2; kn = k+2
      if(isfrozen3d(in,jn,kn)) then; Tr6t3p2 = T(in,jn,kn); else; Tr6t3p2 = INF; endif
   endif
      
   !pixels with a pixel-distance 2 from the center must be lower in value otherwise use other side or first order 
        
   Tm2(1) = second_derivative(Txm1,Txm2,Txp1,Txp2); if(IsInf(Tm2(1))) then; Tm2(1) = 0; else; Order(1) = 2; endif
   Tm2(2) = second_derivative(Tym1,Tym2,Typ1,Typ2); if(IsInf(Tm2(2))) then; Tm2(2) = 0; else; Order(2) = 2; endif
   Tm2(3) = second_derivative(Tzm1,Tzm2,Tzp1,Tzp2); if(IsInf(Tm2(3))) then; Tm2(3) = 0; else; Order(3) = 2; endif
    
   if(usecross) then
      Tm2(4) = Tm2(1); Order(4) = Order(1)
      Tm2(5) = second_derivative(Tr2t2m1,Tr2t2m2,Tr2t2p1,Tr2t2p2); if(IsInf(Tm2(5))) then; Tm2(5) = 0.; else; Order(5) = 2; endif
      Tm2(6) = second_derivative(Tr2t3m1,Tr2t3m2,Tr2t3p1,Tr2t3p2); if(IsInf(Tm2(6))) then; Tm2(6) = 0.; else; Order(6) = 2; endif
 
      Tm2(7) = Tm2(2); Order(7) = Order(2)
      Tm2(8) = second_derivative(Tr3t2m1,Tr3t2m2,Tr3t2p1,Tr3t2p2); if(IsInf(Tm2(8))) then; Tm2(8) = 0.; else; Order(8) = 2; endif
      Tm2(9) = second_derivative(Tr3t3m1,Tr3t3m2,Tr3t3p1,Tr3t3p2); if(IsInf(Tm2(9))) then; Tm2(9) = 0.; else; Order(9) = 2; endif
 
      Tm2(10) = Tm2(3); Order(10) = Order(3)
      Tm2(11) = second_derivative(Tr4t2m1,Tr4t2m2,Tr4t2p1,Tr4t2p2)
      if(IsInf(Tm2(11))) then; Tm2(11) = 0.; else; Order(11) = 2; endif
      Tm2(12) = second_derivative(Tr4t3m1,Tr4t3m2,Tr4t3p1,Tr4t3p2)
      if(IsInf(Tm2(12))) then; Tm2(12) = 0.; else; Order(12) = 2; endif

      Tm2(13) = Tm2(9); Order(13) = Order(9)
      Tm2(14) = second_derivative(Tr5t2m1,Tr5t2m2,Tr5t2p1,Tr5t2p2)
      if(IsInf(Tm2(14))) then; Tm2(14) = 0.; else; Order(14) = 2; endif
      Tm2(15) = second_derivative(Tr5t3m1,Tr5t3m2,Tr5t3p1,Tr5t3p2)
      if(IsInf(Tm2(15))) then; Tm2(15) = 0.; else; Order(15) = 2; endif
            
      Tm2(16) = Tm2(8); Order(16) = Order(8)
      Tm2(17) = second_derivative(Tr6t2m1,Tr6t2m2,Tr6t2p1,Tr6t2p2)
      if(IsInf(Tm2(17))) then; Tm2(17) = 0.; else; Order(17) = 2; endif
      Tm2(18) = second_derivative(Tr6t3m1,Tr6t3m2,Tr6t3p1,Tr6t3p2)
      if(IsInf(Tm2(18))) then; Tm2(18) = 0.; else; Order(18) = 2; endif
  endif
        
endif  
    
! Calculate the distance using x and y direction 
Coeff(1) = 0.; Coeff(2) = 0.; Coeff(3) = -1.0/(max(Fijk**2,eps))   

do p = 1,3
   select case(Order(p))
   case(1)
      Coeff(1) = Coeff(1) + G1(p); Coeff(2) = Coeff(2) -2.0 * Tm(p) * G1(p)
      Coeff(3) = Coeff(3) + (Tm(p)**2) * G1(p)
   case(2)
      Coeff(1) = Coeff(1) + G2(p); Coeff(2) = Coeff(2) - 2.0 * Tm2(p) * G2(p)
      Coeff(3) = Coeff(3) + (Tm2(p)**2) * G2(p)
   end select  
enddo          
    
! compute root
call roots(Coeff,ansroot)    
    
Tt = max(ansroot(1),ansroot(2))          
        
! Calculate the distance using the cross directions 
if(usecross) then

   do q = 1,5
   
      Coeff(1) = Coeff(1) + 0; Coeff(2) = Coeff(2) + 0; Coeff(3) = Coeff(3) - 1.0/max(Fijk**2,eps)
      
      do p = (q*3)+1,(q+1)*3
         select case(Order(p))
         case(1)
            Coeff(1) = Coeff(1) + G1(p); Coeff(2) = Coeff(2) - 2.0 * Tm(p) * G1(p)
            Coeff(3) = Coeff(3) + (Tm(p)**2) * G1(p)
         case(2)
            Coeff(1) = Coeff(1) + G2(p); Coeff(2) = Coeff(2) - 2.0 * Tm2(p) * G2(p)
            Coeff(3) = Coeff(3) + (Tm2(p)**2) * G2(p)
         end select
      enddo
      
      ! Select maximum root solution and minimum distance value of both stensils   
      if(Coeff(1) .gt. 0) then
         call roots(Coeff,ansroot); Tt2 = max(ansroot(1),ansroot(2)); Tt = min(Tt,Tt2)
      endif
      
   enddo           
endif            
        
! Upwind condition check, current distance must be larger then direct neighbours used in solution
if(usecross) then
   do q = 1,18
      if(IsFinite(Tm(q)) .and. (Tt .lt. Tm(q))) then
         Tt = Tm(minarray(Tm,18,size(Tm))) + (1. / (max(Fijk,eps)))   
      endif   
   enddo   

else

   do q = 1,3   
      if(IsFinite(Tm(q)) .and. (Tt .lt. Tm(q))) then
         Tt = Tm(minarray(Tm,3,size(Tm))) + (1. / max(Fijk,eps))              
      endif
   enddo

endif       

END SUBROUTINE calculate_distance

!=========================================================================================

FUNCTION minarray(A,l,n)

use def_kind

implicit none

real(kind=r_single),dimension(n),intent(in)        :: A
integer(kind=i_single),intent(in)                  :: l,n
integer(kind=i_single)                             :: minarray

!---------------------------------------------------

minarray = minloc(A(1:l),dim=1)

END FUNCTION minarray

!=========================================================================================

FUNCTION iszero(a)

use def_kind; use fast_marching, only: eps

implicit none

real(kind=r_single),intent(in) :: a
integer(kind=i_single)         :: iszero 

!-------------------------------------------

iszero = 0

if (a**2 .lt. eps) iszero = 1 

END FUNCTION iszero

!=========================================================================================

FUNCTION isnotzero(a)

use def_kind; use fast_marching, only: eps

implicit none

real(kind=r_single),intent(in) :: a
integer(kind=i_single)         :: isnotzero 

!-------------------------------------------

isnotzero = 0

if (a**2 .gt. eps) isnotzero = 1 

END FUNCTION isnotzero

!=========================================================================================

SUBROUTINE roots(Coeff,ans)

use def_kind

implicit none

real(kind=r_single),dimension(3),intent(in) :: Coeff
real(kind=r_single),dimension(2),intent(out):: ans
real(kind=r_single)                         :: a,b,c,r1,r2,d
integer(kind=i_single)                      :: isnotzero

!-------------------------------------------------------------------

a = Coeff(1); b = Coeff(2); c = Coeff(3)

d = max((b**2) - 4.0 * a *c, 0.)

if(isnotzero(a) .eq. 1) then

   ans(1) = (-b - sqrt(d)) / (2.0 * a)
   ans(2) = (-b + sqrt(d)) / (2.0 * a)

else

   r1 = (-b - sqrt(d))
   r2 = (-b + sqrt(d))
   
   if(isnotzero(r1) .eq. 1) then

      if(isnotzero(r2) .eq. 1) then
         ans(1) = (2.0 * c) / r1; ans(2) = (2.0 * c) / r2
      else   
         ans(1) = (2.0 * c) / r1; ans(2) = (2.0 * c) / r1
      endif

   elseif(isnotzero(r2) .eq. 1) then      

      ans(1) =  (2.0 * c) / r2; ans(2) =  (2.0 * c) / r2  

   else   
    
      ans(1) = 0; ans(2) = 0
      
   endif
   
endif

END SUBROUTINE roots

!=========================================================================================      
    
FUNCTION IsFinite(x)

use def_kind; use fast_marching, only: INF

implicit none

real(kind=r_single),intent(in) :: x
logical                        :: IsFinite

!-----------------------------------------------------------------

IsFinite = .FALSE.

if( (x .lt. INF) .and. (x .gt. -INF) ) IsFinite = .TRUE. 

END FUNCTION IsFinite

!=========================================================================================

FUNCTION IsInf(x)

use def_kind; use fast_marching, only: INF

implicit none

real(kind=r_single),intent(in) :: x
logical                        :: IsInf

!-----------------------------------------------------------------

IsInf = .FALSE.

if(x .eq. INF) IsInf = .TRUE. 

END FUNCTION IsInf

!=========================================================================================

FUNCTION isntfrozen3d(i,j,k)

use def_kind; use fast_marching, only: dims,Frozen

implicit none

integer(kind=i_single),intent(in):: i,j,k 
logical                          :: isntfrozen3d

!----------------------------------------------------------------- 

isntfrozen3d = .FALSE.

! if statement split to avoid out-of-bound index for Frozen
if( (i .ge. 1) .and. (j .ge. 1) .and. (k .ge. 1) .and. (i .le. dims(1)) .and. (j .le. dims(2)) .and. &
&   (k .le. dims(3))) then
   if(Frozen(i,j,k) .ne. 2) isntfrozen3d = .TRUE.
endif

END FUNCTION isntfrozen3d

!=========================================================================================

FUNCTION isfrozen3d(i,j,k)

use def_kind; use fast_marching, only: dims,Frozen

implicit none

integer(kind=i_single),intent(in):: i,j,k 
logical                          :: isfrozen3d

!----------------------------------------------------------------- 

isfrozen3d = .FALSE.

! if statement split to avoid out-of-bound index for Frozen
if( (i .ge. 1) .and. (j .ge. 1) .and. (k .ge. 1) .and. (i .le. dims(1)) .and. (j .le. dims(2)) .and. &
&   (k .le. dims(3))) then
   if(Frozen(i,j,k) .eq. 2) isfrozen3d = .TRUE.
endif

END FUNCTION isfrozen3d

!=========================================================================================

