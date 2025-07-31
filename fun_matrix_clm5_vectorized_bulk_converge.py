import time
import numpy as np
import torch
import traceback
import math


def fun_bulk_simu(tensor_para, tensor_frocing_steady_state):
	"""
	Runs the same CLM5 model as fun_model_prediction() in fun_matrix_clm5_vectorized.py,
	but returns more variables describing the underlying processes, instead of just predicted SOC.
	"""
	device = tensor_para.device
	para = tensor_para
	frocing_steady_state = tensor_frocing_steady_state 

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)
	

	n_soil_layer = 20

	# Initialize the final outputs to store the simulation results: carbon_input_sum, cpool_steady_state, cpools_layer, soc_layer, 
	# residence_time, total_res_time_base, res_time_base_pools, t_scaaler, bulk_Aler, w_sc, bulk_K, bulk_V, bulk_xi, bulk_I, litter_fraction
	profile_num = para.shape[0]
	carbon_input = (torch.ones((profile_num, 1))*np.nan).to(device)
	cpool_steady_state = (torch.ones((profile_num, 140))*np.nan).to(device)
	cpools_layer = (torch.ones((profile_num, 20))*np.nan).to(device)
	soc_layer = (torch.ones((profile_num, 20))*np.nan).to(device)
	total_res_time = (torch.ones((profile_num, 20))*np.nan).to(device)
	total_res_time_base = (torch.ones((profile_num, 20))*np.nan).to(device)
	res_time_base_pools = (torch.ones((profile_num, 140))*np.nan).to(device)
	t_scaler = (torch.ones((profile_num, 20))*np.nan).to(device)
	bulk_A = (torch.ones((profile_num, 1))*np.nan).to(device)
	w_scaler = (torch.ones((profile_num, 20))*np.nan).to(device)
	bulk_K = (torch.ones((profile_num, 1))*np.nan).to(device)
	bulk_V = (torch.ones((profile_num, 1))*np.nan).to(device)
	bulk_xi = (torch.ones((profile_num, 1))*np.nan).to(device)
	bulk_I = (torch.ones((profile_num, 1))*np.nan).to(device)
	litter_fraction = (torch.ones((profile_num, 1))*np.nan).to(device)
	# calculate soc solution for each profile
	for iprofile in range(0, profile_num):
		profile_para = para[iprofile, :]
		profile_force_steady_state = frocing_steady_state[iprofile, :, :, :]
		# profile_obs_layer_depth = obs_layer_depth[iprofile, :]
		# valid_layer_loc = torch.where(torch.isnan(profile_obs_layer_depth) == False)[0]

		if torch.isnan(torch.sum(profile_para)) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:12, 0, 1:8])) == False and \
			torch.isnan(torch.sum(profile_force_steady_state[0:20, 0:12, 8:13])) == False:
			
			# print(profile_para)
			# model simulation
			carbon_input_sum_profile, cpool_steady_state_profile, cpools_layer_profile, soc_layer_profile, \
			total_res_time_profile, total_res_time_base_profile, res_time_base_pools_profile, t_scaler_profile, \
			bulk_A_profile, w_scaler_profile, bulk_K_profile, bulk_V_profile, bulk_xi_profile, bulk_I_profile, \
			litter_fraction_profile = fun_matrix_clm5(profile_para, profile_force_steady_state)

			# print the shape of the outputs
			# print("carbon_input_sum_profile", carbon_input_sum_profile.shape)
			# print("cpool_steady_state_profile", cpool_steady_state_profile.shape)
			# print("cpools_layer_profile", cpools_layer_profile.shape)
			# print("soc_layer_profile", soc_layer_profile.shape)
			# print("total_res_time_profile", total_res_time_profile.shape)
			# print("total_res_time_base_profile", total_res_time_base_profile.shape)
			# print("res_time_base_pools_profile", res_time_base_pools_profile.shape)
			# print("t_scaler_profile", t_scaler_profile.shape)
			# print("bulk_A_profile", bulk_A_profile.shape)
			# print("w_scaler_profile", w_scaler_profile.shape)
			# print("bulk_K_profile", bulk_K_profile.shape)
			# print("bulk_V_profile", bulk_V_profile.shape)
			# print("bulk_xi_profile", bulk_xi_profile.shape)
			# print("bulk_I_profile", bulk_I_profile.shape)
			# print("litter_fraction_profile", litter_fraction_profile.shape)

			
			# store the results
			carbon_input[iprofile, :] = carbon_input_sum_profile
			cpool_steady_state[iprofile, :] = cpool_steady_state_profile.squeeze()
			cpools_layer[iprofile, :] = cpools_layer_profile
			soc_layer[iprofile, :] = soc_layer_profile
			total_res_time[iprofile, :] = total_res_time_profile
			total_res_time_base[iprofile, :] = total_res_time_base_profile
			res_time_base_pools[iprofile, :] = res_time_base_pools_profile.squeeze()
			t_scaler[iprofile, :] = t_scaler_profile
			bulk_A[iprofile, :] = bulk_A_profile
			w_scaler[iprofile, :] = w_scaler_profile
			bulk_K[iprofile, :] = bulk_K_profile
			bulk_V[iprofile, :] = bulk_V_profile
			bulk_xi[iprofile, :] = bulk_xi_profile
			bulk_I[iprofile, :] = bulk_I_profile
			litter_fraction[iprofile, :] = litter_fraction_profile

			# end for
		# end if 
	#end for iprofile
	return carbon_input, cpool_steady_state, cpools_layer, soc_layer, total_res_time, total_res_time_base, res_time_base_pools, t_scaler, bulk_A, w_scaler, bulk_K, bulk_V, bulk_xi, bulk_I, litter_fraction
	
# end def fun_model_simu

#######################################################
# forward simulation for clm5
#######################################################
def fun_matrix_clm5(para, frocing_steady_state):
	device = para.device
	#---------------------------------------------------
	# offical starting simulation
	#---------------------------------------------------
	use_beta = 1
	normalize_q10_to_century_tfunc = False
	global timestep_num, season_num, month_num, \
		kelvin_to_celsius, use_vertsoilc, npool, npool_vr, n_soil_layer, \
		days_per_year, secspday, days_per_month, days_per_season, days_per_timestep
		
	season_num = 4
	month_num = 12
	kelvin_to_celsius = 273.15
	use_vertsoilc = 1
	npool = 7
	npool_vr = 140
	n_soil_layer = 20
	days_per_year = 365
	secspday = 24*60*60
	days_per_month = torch.tensor([[31, 30, 31, 28, 31, 30, 31, 31, 30, 31, 30, 31]]).transpose(0, 1).to(device)
	days_per_season = torch.tensor([[92, 89, 92, 92]]).transpose(0, 1).to(device)
	
	timestep_num = month_num
	days_per_timestep = days_per_season

	global max_altdepth_cryoturbation, max_depth_cryoturb
	max_altdepth_cryoturbation = 2
	max_depth_cryoturb = 3

	global dz, zisoi, zisoi_0, zsoi, dz_node
	# width between two interfaces
	dz = torch.tensor([2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, \
		8.000000000000000E-002, 0.120000000000000, 0.160000000000000, \
		0.200000000000000, 0.240000000000000, 0.280000000000000, \
		0.320000000000000, 0.360000000000000, 0.400000000000000, \
		0.440000000000000, 0.540000000000000, 0.640000000000000, \
		0.740000000000000, 0.840000000000000, 0.940000000000000, \
		1.04000000000000, 1.14000000000000, 2.39000000000000, \
		4.67553390593274, 7.63519052838329, 11.1400000000000, \
		15.1154248593737]).to(device)
	
	# depth of the interface
	zisoi = torch.tensor([2.000000000000000E-002, 6.000000000000000E-002, \
		0.120000000000000, 0.200000000000000, 0.320000000000000, \
		0.480000000000000, 0.680000000000000, 0.920000000000000, \
		1.20000000000000, 1.52000000000000, 1.88000000000000, \
		2.28000000000000, 2.72000000000000, 3.26000000000000, \
		3.90000000000000, 4.64000000000000, 5.48000000000000, \
		6.42000000000000, 7.46000000000000, 8.60000000000000, \
		10.9900000000000, 15.6655339059327, 23.3007244343160, \
		34.4407244343160, 49.5561492936897]).to(device)

	zisoi_0 = 0;

	# depth of the node                                                   
	zsoi = torch.tensor([1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, \
		0.160000000000000, 0.260000000000000, 0.400000000000000, \
		0.580000000000000, 0.800000000000000, 1.06000000000000, \
		1.36000000000000, 1.70000000000000, 2.08000000000000, \
		2.50000000000000, 2.99000000000000, 3.58000000000000, \
		4.27000000000000, 5.06000000000000, 5.95000000000000, \
		6.94000000000000, 8.03000000000000, 9.79500000000000, \
		13.3277669529664, 19.4831291701244, 28.8707244343160, \
		41.9984368640029]).to(device)

	# depth between two node
	dz_node = zsoi - torch.cat((torch.tensor([0.0]).to(device), zsoi[:-1]), axis = 0)

	# construct a diagonal matrix that contains dz for each layer (20 layers) and 7 pools
	dz_matrix = torch.diag(-1*torch.ones(npool_vr)).to(device)
	# fill the diagonal matrix with dz for each pool (7 pools)
	dz_matrix.diagonal()[0:20] = dz[0:20]
	dz_matrix.diagonal()[20:40] = dz[0:20]
	dz_matrix.diagonal()[40:60] = dz[0:20]
	dz_matrix.diagonal()[60:80] = dz[0:20]
	dz_matrix.diagonal()[80:100] = dz[0:20]
	dz_matrix.diagonal()[100:120] = dz[0:20]
	dz_matrix.diagonal()[120:140] = dz[0:20]
	dz_matrix_diagonal = dz_matrix.diagonal().view(npool_vr, 1)


	#---------------------------------------------------
	# steady state forcing
	#---------------------------------------------------
	input_vector_cwd_steady_state = frocing_steady_state[0:12, 0, 1]
	input_vector_litter1_steady_state = frocing_steady_state[0:12, 0, 2]
	input_vector_litter2_steady_state = frocing_steady_state[0:12, 0, 3]
	input_vector_litter3_steady_state = frocing_steady_state[0:12, 0, 4]
	altmax_lastyear_profile_steady_state = frocing_steady_state[0:12, 0, 5]
	altmax_current_profile_steady_state = frocing_steady_state[0:12, 0, 6]
	nbedrock_steady_state = frocing_steady_state[0:12, 0, 7].type(torch.int)
	
	xio_steady_state = frocing_steady_state[0:20, 0:12, 8]
	xin_steady_state = frocing_steady_state[0:20, 0:12, 9]
	sand_vector_steady_state = frocing_steady_state[0:20, 0:12, 10]
	soil_temp_profile_steady_state = frocing_steady_state[0:20, 0:12, 11]
	soil_water_profile_steady_state = frocing_steady_state[0:20, 0:12, 12]
	
	#---------------------------------------------------
	# define parameters to be optimised
	#---------------------------------------------------
	# diffusion (bioturbation) 10^(-4) (m2/yr)
	bio = para[0]*(5*1e-4 - 3*1e-5) + 3*1e-5
	# cryoturbation 5*10^(-4) (m2/yr)
	cryo = para[1]*(16*1e-4 - 3*1e-5) + 3*1e-5

	#  Q10 (unitless) 1.5
	q10 = para[2]*(3 - 1.2) + 1.2
	# Q10 when forzen (unitless) 1.5
	fq10 = q10
	# parameters used in vertical discretization of carbon inputs 10 (metre)
	# efolding = para[3]*(1 - 0.0001) + 0.0001
	efolding = para[3]*(1 - 0.1) + 0.1
	# turnover time of CWD (yr) 3.3333
	tau4cwd = para[4]*(6 - 1) + 1
	# tau for metabolic litter (yr) 0.0541
	tau4l1 = para[5]*(0.11 - 0.0001) + 0.0001
	# tau for cellulose litter (yr) 0.2041
	tau4l2 = para[6]*(0.3 - 0.1) + 0.1
	# tau for lignin litter (yr)
	tau4l3 = tau4l2
	# tau for fast SOC (yr) 0.1370
	tau4s1 = para[7]*(0.5 - 0.0001) + 0.0001
	# tau for slow SOC (yr) 5
	tau4s2 = para[8]*(10 - 1) + 1
	# tau for passive SOC (yr) 222.222
	tau4s3 = para[9]*(400 - 20) + 20
	# fraction from l1 to s2, 0.45
	fl1s1 = para[10]*(0.8 - 0.1) + 0.1
	# fraction from l2 to s1, 0.5
	fl2s1 = para[11]*(0.8 - 0.2) + 0.2
	# fraction from l3 to s2, 0.5
	fl3s2 = para[12]*(0.8 - 0.2) + 0.2
	# fraction from s1 to s2, sand dependeted
	fs1s2 = para[13]*(0.4 - 0.0001) + 0.0001
	# fraction from s1 to s3, sand dependeted
	fs1s3 = para[14]*(0.1 - 0.0001) + 0.0001
	# fraction from s2 to s1, 0.42
	fs2s1 = para[15]*(0.74 - 0.1) + 0.1
	# fraction from s2 to s3, 0.03
	fs2s3 = para[16]*(0.1 - 0.0001) + 0.0001
	# fraction from s3 to s1, 0.45
	fs3s1 = para[17]*(0.9 - 0.0001) + 0.0001
	# fraction from cwd to l2, 0.76
	fcwdl2 = para[18]*(1 - 0.5) + 0.5
	
	# water scaling factor
	w_scaling = para[19]*(5 - 0.0001) + 0.0001
	# beta to describe the shape of vertical profile
	# beta = 0.95
	# or fix it at first ~ 0.6/0.7
	# beta = para[20]*(0.9 - 0.5) + 0.5
	beta = para[20] *(0.9999 - 0.5) + 0.5
	# beta = 0.7 *(0.9 - 0.5) + 0.5
	# beta = 0.8


	# maximum and minimum water potential (MPa)
	maxpsi= -0.0020
	minpsi= -2; # minimum water potential (MPa)
	adv = 0 # parameter for advection (m/yr)

	#####################################################
	# steady state solutions
	#####################################################

	#---------------------------------------------------
	# steady state env scalar
	#---------------------------------------------------
	xiw = (torch.ones(n_soil_layer, timestep_num)*np.nan).to(device)
	xio = xio_steady_state
	xin = xin_steady_state

	xit_above_freezing = torch.pow(q10, ((soil_temp_profile_steady_state - (kelvin_to_celsius + 25))/10))  # Above freezing case first
	xit_below_freezing = torch.pow(q10, ((273.15 - 298.15)/10)) * torch.pow(fq10, ((soil_temp_profile_steady_state - (0 + kelvin_to_celsius))/10))	
	freezing_mask = (soil_temp_profile_steady_state < (0 + kelvin_to_celsius)).detach().int()  # Create a mask which is True when the soil temperatue is below freezing
	xit = xit_above_freezing * (1-freezing_mask) + xit_below_freezing * freezing_mask  # [freezing_mask] = xit_below_freezing[freezing_mask].clone()
	catanf_30 = catanf(torch.tensor(30.0).to(device))
	normalization_tref = torch.tensor(15).to(device)
	if normalize_q10_to_century_tfunc == True:
		# scale all decomposition rates by a constant to compensate for offset between original CENTURY temp func and Q10
		normalization_factor = (catanf(normalization_tref)/catanf_30) / (q10**((normalization_tref-25)/10))
		xit = xit * normalization_factor


	xiw = soil_water_profile_steady_state*w_scaling
	xiw[xiw > 1] = 1

	#---------------------------------------------------
	# steady state tridiagnal matrix, A matrix, K matrix, fire matrix
	#---------------------------------------------------
	sand_vector = torch.mean(sand_vector_steady_state, axis = 1)

	# allocation matrix (horizontal transfers within same layer)
	a_ma = a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector)

	kk_ma_middle = (torch.zeros([npool_vr, npool_vr, timestep_num])*np.nan).to(device) 
	tri_ma_middle = (torch.zeros([npool_vr, npool_vr, timestep_num])*np.nan).to(device) 
	
	for itimestep in range(timestep_num):
		# decomposition matrix
		timesteply_xit = xit[:, itimestep]
		timesteply_xiw = xiw[:, itimestep]
		timesteply_xio = xio[:, itimestep]
		timesteply_xin = xin[:, itimestep]

		# K matrix
		kk_ma = kk_matrix_vectorized(timesteply_xit, timesteply_xiw, timesteply_xio, timesteply_xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3)
		kk_ma_middle[:, :, itimestep] = kk_ma

		# tri matrix	
		timesteply_nbedrock = nbedrock_steady_state[itimestep]
		timesteply_altmax_current_profile = altmax_current_profile_steady_state[itimestep]
		timesteply_altmax_lastyear_profile = altmax_lastyear_profile_steady_state[itimestep]


		tri_ma = tri_matrix_old_improved(timesteply_nbedrock, timesteply_altmax_current_profile, timesteply_altmax_lastyear_profile, bio, adv, cryo)
		tri_ma_middle[:, :, itimestep] = tri_ma

	# end for itimestep
	tri_ma = torch.mean(tri_ma_middle, axis = 2)
	kk_ma = torch.mean(kk_ma_middle, axis = 2)
	
	#---------------------------------------------------
	# steady state vertical profile, input allocation
	#---------------------------------------------------
	# in the original beta model in Jackson et al 1996, the unit for the depth of the soil is cm (dmax*100)
	m_to_cm = 100

	vertical_prof = (torch.ones(n_soil_layer)*np.nan).to(device) 
	if torch.mean(altmax_lastyear_profile_steady_state) > 0:
		# New way to calculate vertical_prof (vectorized)
		vertical_prof = (torch.ones(n_soil_layer)*np.nan).to(device) 
		vertical_prof[0] = (beta**((zisoi_0)*m_to_cm) - beta**(zisoi[0]*m_to_cm))/dz[0]
		vertical_prof[1:n_soil_layer] = (beta**((zisoi[0:n_soil_layer-1])*m_to_cm) - beta**(zisoi[1:n_soil_layer]*m_to_cm))/dz[1:n_soil_layer]


	else:
		vertical_prof[0] = 1/dz[0]
		vertical_prof[1:] = 0
	# end if np.mean(altmax_lastyear_profile_steady_state) > 0:

	vertical_input = dz[0:n_soil_layer]*vertical_prof/sum(vertical_prof*dz[0:n_soil_layer])
	
	#---------------------------------------------------
	# steady state analytical solution of soc
	#---------------------------------------------------
	matrix_in = (torch.ones([npool_vr, 1])*np.nan).to(device) 
	# total input amount
	input_tot_cwd = torch.nansum(input_vector_cwd_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter1 = torch.nansum(input_vector_litter1_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter2 = torch.nansum(input_vector_litter2_steady_state)/days_per_year # (gc/m2/day)
	input_tot_litter3 = torch.nansum(input_vector_litter3_steady_state)/days_per_year # (gc/m2/day)

	# redistribution by beta
	matrix_in[0:20, 0] = input_tot_cwd*vertical_input/dz[0:n_soil_layer] # litter input gc/m3/day
	matrix_in[20:40, 0] = input_tot_litter1*vertical_input/dz[0:n_soil_layer]
	matrix_in[40:60, 0] = input_tot_litter2*vertical_input/dz[0:n_soil_layer]
	matrix_in[60:80, 0] = input_tot_litter3*vertical_input/dz[0:n_soil_layer]
	matrix_in[80:140, 0] = 0

	# calculate the sum of the input
	carbon_input_sum = (input_tot_cwd + input_tot_litter1 + input_tot_litter2 + input_tot_litter3) * days_per_year # unit gc/m2/yr
	
	# analytical solution of soc pools
	try:
		cpool_steady_state = torch.linalg.solve((torch.matmul(a_ma, kk_ma)- tri_ma), (-matrix_in))
	except Exception:
		traceback.print_exc()
		print("Predicted Parameters in Bulk simu cpool: ", para)
		# check if the matrix is singular and print the matrix
		# check a_ma
		if torch.isnan(torch.sum(a_ma)):
			print("a_ma contains nan")
		if torch.det(a_ma) == 0:
			print("a_ma is singular")
		# check kk_ma
		if torch.isnan(torch.sum(kk_ma)):
			print("kk_ma contains nan")
		if torch.det(kk_ma) == 0:
			print("kk_ma is singular")
			print(torch.diagonal(kk_ma))
		# check tri_ma
		if torch.isnan(torch.sum(tri_ma)):
			print("tri_ma contains nan")
		if torch.det(tri_ma) == 0:
			print("tri_ma is singular")
			print(torch.diagonal(tri_ma, offset=0))
			print(torch.diagonal(tri_ma, offset=1))
			print(torch.diagonal(tri_ma, offset=-1))
		# check matrix_in
		if torch.isnan(torch.sum(matrix_in)):
			print("matrix_in contains nan")
		if torch.det(torch.matmul(a_ma, kk_ma)-tri_ma) == 0:
			print("a_ma*kk_ma - tri_ma is singular")
		cpool_steady_state = (torch.ones([140, 1])*(-1.0)).to(device)*torch.sum(para)/torch.sum(para)
	# end try

	# convert cpools_steady_state to a vector with size [20]
	cpools_layer = torch.cat((cpool_steady_state[0:20, :], cpool_steady_state[20:40, :], cpool_steady_state[40:60, :], cpool_steady_state[60:80, :], cpool_steady_state[80:100, :], cpool_steady_state[100:120, :], cpool_steady_state[120:140, :]), dim = 1)	
	cpools_layer = torch.sum(cpools_layer, axis = 1) # unit gC/m3
	soc_layer = torch.cat((cpool_steady_state[80:100, :], cpool_steady_state[100:120, :], cpool_steady_state[120:140, :]), dim = 1)
	soc_layer = torch.sum(soc_layer, axis = 1) # unit gC/m3

	
	#--------------------------------------------------- Residence Time ---------------------------------------------------
	# Initialize the B matrix
	B_matrix = (torch.ones([npool_vr, 1])*np.nan).to(device)

	# Calculate the B matrix
	B_matrix[0:20, 0] = (input_tot_cwd * vertical_input) / (carbon_input_sum * dz[0:n_soil_layer])
	B_matrix[20:40, 0] = (input_tot_litter1 * vertical_input) / (carbon_input_sum * dz[0:n_soil_layer])
	B_matrix[40:60, 0] = (input_tot_litter2 * vertical_input) / (carbon_input_sum * dz[0:n_soil_layer])
	B_matrix[60:80, 0] = (input_tot_litter3 * vertical_input) / (carbon_input_sum * dz[0:n_soil_layer])
	B_matrix[80:140, 0] = 0

	# diagonal scaling matrix
	diag_scaler_monthly = (torch.ones([n_soil_layer, month_num])*np.nan).to(device)
	for imonth in range(month_num):
		diag_scaler_monthly[:, imonth] = 1/(xiw[:, imonth] * xit[:, imonth] * xio[:, imonth] * xin[:, imonth])
	# end for imonth
	t_scaler = torch.nanmean(xit, axis = 1).to(device)
	w_scaler = torch.nanmean(xiw, axis = 1).to(device)


	diag_scaler_temp = torch.nanmean(diag_scaler_monthly, axis = 1).to(device)
	diag_scaler = torch.cat([diag_scaler_temp] * 7)
	diag_scaler = torch.diag(diag_scaler)
	# print("diag_scalar", diag_scalar)
	# print("diag_scalar", diag_scalar.shape)

	# diag_scaler = (torch.ones([npool_vr, npool_vr])*0).to(device)
	# # Fill the diagonal of diag_scaler with diag_scaler_temp for each pool
	# diag_scaler[0:20, 0:20] = torch.diag(diag_scaler_temp[0:20])
	# diag_scaler[20:40, 20:40] = torch.diag(diag_scaler_temp[0:20])
	# diag_scaler[40:60, 40:60] = torch.diag(diag_scaler_temp[0:20])
	# diag_scaler[60:80, 60:80] = torch.diag(diag_scaler_temp[0:20])
	# diag_scaler[80:100, 80:100] = torch.diag(diag_scaler_temp[0:20])
	# diag_scaler[100:120, 100:120] = torch.diag(diag_scaler_temp[0:20])
	# diag_scaler[120:140, 120:140] = torch.diag(diag_scaler_temp[0:20])

	# calculate the residence time
	residence_time = torch.linalg.solve((torch.matmul(a_ma, kk_ma)- tri_ma), (-B_matrix))
	try: 
		residence_time_baseline = torch.linalg.solve((torch.matmul(a_ma, kk_ma) - tri_ma)*diag_scaler, (-B_matrix))
	except Exception:
		traceback.print_exc()
		print("Predicted Parameters in Bulk simu residence time: ", para)
		# check if the matrix is singular and print the matrix
		# check a_ma
		if torch.isnan(torch.sum(a_ma)):
			print("a_ma contains nan")
		if torch.det(a_ma) == 0:
			print("a_ma is singular")
		# check kk_ma
		if torch.isnan(torch.sum(kk_ma)):
			print("kk_ma contains nan")
		if torch.det(kk_ma) == 0:
			print("kk_ma is singular")
			print(torch.diagonal(kk_ma))
		# check tri_ma
		if torch.isnan(torch.sum(tri_ma)):
			print("tri_ma contains nan")
		if torch.det(tri_ma) == 0:
			print("tri_ma is singular")
			print(torch.diagonal(tri_ma, offset=0))
			print(torch.diagonal(tri_ma, offset=1))
			print(torch.diagonal(tri_ma, offset=-1))
		# check matrix_in
		if torch.isnan(torch.sum(B_matrix)):
			print("B_matrix contains nan")
		if torch.det(B_matrix) == 0:
			print("B_matrix is singular")
			print(B_matrix)
		if torch.isnan(torch.sum(diag_scaler)):
			print("diag_scaler contains nan")
		if torch.det(diag_scaler) == 0:
			print("diag_scaler is singular")
			print(diag_scaler)		
		if torch.det(torch.matmul(a_ma, kk_ma)-tri_ma) == 0:
			print("a_ma*kk_ma - tri_ma is singular")
		if torch.det(torch.matmul(torch.matmul(a_ma, kk_ma) - tri_ma, diag_scaler)) == 0:
			print("a_ma*kk_ma - tri_ma is singular")
		residence_time_baseline = (torch.ones([140, 1])*(-1.0)).to(device)*torch.sum(para)/torch.sum(para)

	total_res_time = torch.cat((residence_time[80:100, :], residence_time[100:120, :], residence_time[120:140, :]), dim = 1)
	total_res_time = torch.sum(total_res_time, axis = 1)

	total_res_time_base = torch.cat((residence_time_baseline[80:100, :], residence_time_baseline[100:120, :], residence_time_baseline[120:140, :]), dim = 1)
	total_res_time_base = torch.sum(total_res_time_base, axis = 1)

	res_time_base_pools = residence_time_baseline

	#--------------------------------------------------- Bulk Process ---------------------------------------------------
	## I matrix ##
	cum_fraction_input = torch.cumsum(vertical_input, dim = 0).to(device)
	# for the value over 1, set it to nan
	cum_fraction_input[cum_fraction_input >= 1] = np.nan
	bulk_I = torch.nanmean(torch.exp(torch.log(1 - cum_fraction_input)/(zsoi[0:n_soil_layer]*100)), axis = 0).to(device)

	## K matrix ##
	# calculate the sum of cpool_steady_state
	cpools_total = torch.stack([torch.sum(cpool_steady_state[0:20, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(cpool_steady_state[20:40, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(cpool_steady_state[40:60, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(cpool_steady_state[60:80, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(cpool_steady_state[80:100, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(cpool_steady_state[100:120, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(cpool_steady_state[120:140, :] * dz[0:n_soil_layer], dim = 0)], dim = 0).to(device)
	# print("cpools_total", cpools_total.shape)
	# calculate decomposition rate
	decom_cpools = torch.tensor([tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3]).reciprocal().unsqueeze(1).to(device)  # , dtype=torch.float32
	# print("tau4cwd", tau4cwd)
	# print("decom_cpools", decom_cpools.shape)
	# print("decom_cpools", decom_cpools.shape)
	# print("cpools_total", cpools_total.shape)
	# calculate bulk_K
	bulk_K = torch.sum(decom_cpools * (cpools_total / torch.sum(cpools_total))).to(device)

	## litter fraction ##
	decom_cpool_apparent = torch.matmul(kk_ma, cpool_steady_state).to(device)
	# print("decom_cpool_apparent", decom_cpool_apparent.shape)
	total_decom_litter = torch.stack([torch.sum(decom_cpool_apparent[20:40, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(decom_cpool_apparent[40:60, :] * dz[0:n_soil_layer], dim = 0),
		torch.sum(decom_cpool_apparent[60:80, :] * dz[0:n_soil_layer], dim = 0)], dim = 0).to(device)
	# dz_repeated = dz[0:n_soil_layer].repeat(3, 1).t() 
	# total_decom_litter = decom_cpool_apparent[20:80] * dz_repeated
	resp_vector = -torch.sum(a_ma, dim=0).t()
	litter_resp_vector = torch.stack((resp_vector[20:40], resp_vector[40:60], resp_vector[60:80]), dim = 0).t().to(device)

	# convert total_decom_litter to an 1d matrix
	# print("total_decom_litter", total_decom_litter.shape)
	# print("resp_vector", resp_vector.shape)
	# print("litter_resp_vector", litter_resp_vector.shape)
	# total_decom_litter = total_decom_litter.reshape(1, -1)
	# print("total_decom_litter_reshape", total_decom_litter.shape)
	total_resp_litter = torch.matmul(total_decom_litter, litter_resp_vector).to(device)
	litter_fraction = 1 - torch.nansum(total_resp_litter) / torch.nansum(total_decom_litter)
	
	## xi ##
	depth_scaler = torch.exp(-zsoi[:n_soil_layer] / efolding).to(device)
	bulk_xi_layer = (t_scaler * w_scaler * depth_scaler).to(device)
	bulk_xi = (torch.sum(bulk_xi_layer * (soc_layer * dz[:n_soil_layer])) / torch.sum(soc_layer * dz[:n_soil_layer])).to(device)
	if not torch.isreal(bulk_xi):
		bulk_xi = torch.nan
	
	## A matrix ##
	donor_pool_layer = torch.stack([cpool_steady_state[0:20, :], cpool_steady_state[20:40, :], cpool_steady_state[40:60, :], cpool_steady_state[60:80, :], cpool_steady_state[80:100, :], cpool_steady_state[100:120, :], cpool_steady_state[120:140, :]], dim = 1).to(device)
	cue_cpool = torch.tensor([fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1]).to(device)
	# print("cue_cpool", len(cue_cpool))
	donor_pool_size = donor_pool_layer[:, [1, 2, 3, 4, 4, 5, 5, 6]]
	# print("donor_pool_size", donor_pool_size.shape)
	# print("fl1s1", fl1s1.shape)
	# decom_cpool = 1.0 / np.array([tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3])
	# donor_decomp = decom_cpool[[1, 2, 3, 4, 4, 5, 5, 6]]
	# print("decom_cpools", decom_cpools.shape)
	donor_decomp = torch.stack((decom_cpools[1], decom_cpools[2], decom_cpools[3], decom_cpools[4], decom_cpools[4], decom_cpools[5], decom_cpools[5], decom_cpools[6]), dim = 0).to(device)

	# Calculate total_doner_flow
	total_doner_flow = torch.nan * torch.ones(len(cue_cpool), device=device)
	for idoner in range(len(cue_cpool)):
		# print("donor pool size", donor_pool_size[:, idoner].shape)
		# print("donor decomp", donor_decomp[idoner].shape)
		# print("bulk_xi_layer", bulk_xi_layer.shape)
		# print("dz", dz[:n_soil_layer].shape)
		total_doner_flow[idoner] = torch.sum(torch.matmul(donor_pool_size[:, idoner], donor_decomp[idoner]) * bulk_xi_layer * dz[:n_soil_layer])

	bulk_A = torch.sum(cue_cpool * total_doner_flow) / torch.sum(total_doner_flow[[0, 1, 2, 3, 5, 7]])

	## V matrix ##
	bulk_V_monthly = torch.nan * torch.ones(month_num)
	for imonth in range(month_num):
		bulk_V_middle = torch.diag(tri_ma_middle[20:40, 20:40, imonth]) * days_per_year
		bulk_V_monthly[imonth] = torch.sum(bulk_V_middle * (soc_layer * dz[:n_soil_layer]) / torch.sum(soc_layer * dz[:n_soil_layer]))
	bulk_V = torch.nanmean(bulk_V_monthly)


	
	# if soc_layer[-1] > soc_layer[0]:
	# 	soc_layer =  (torch.ones(20)*(-9999.*3))
	# # end if soc_layer[-1] > soc_layer[0]:
	
	outcome = soc_layer

	# check the shape of all the returned variables
	# print("shape of carbon_input_sum", carbon_input_sum.shape)
	# print("shape of cpool_steady_state", cpool_steady_state.shape)
	# print("shape of cpools_layer", cpools_layer.shape)
	# print("shape of soc_layer", soc_layer.shape)
	# print("shape of total_res_time", total_res_time.shape)
	# print("shape of total_res_time_base", total_res_time_base.shape)
	# print("shape of res_time_base_pools", res_time_base_pools.shape)
	# print("shape of t_scaler", t_scaler.shape)
	# print("shape of w_scaler", w_scaler.shape)
	# print("shape of bulk_A", bulk_A.shape)
	# print("shape of bulk_K", bulk_K.shape)
	# print("shape of bulk_V", bulk_V.shape)
	# print("shape of bulk_xi", bulk_xi.shape)
	# print("shape of bulk_I", bulk_I.shape)
	# print("shape of litter_fraction", litter_fraction.shape)

	return carbon_input_sum, cpool_steady_state, cpools_layer, soc_layer, total_res_time, total_res_time_base, res_time_base_pools, t_scaler, bulk_A, w_scaler, bulk_K, bulk_V, bulk_xi, bulk_I, litter_fraction
	
#end def fun_forward_simu_clm5



####################################################################
# TODO: These functions are the same as in fun_matrix_clm5_vectorized.py.
# However, because they use global variables defined above, we cannot just
# call the functions there. It would be nice to refactor this.
#####################################################################

##################################################
# sub-function in matrix equation
##################################################

def a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector):
	device = fl1s1.device
	nlevdecomp = n_soil_layer
	nspools = npool
	nspools_vr = npool_vr
	# a diagnal matrix
	a_ma_vr = torch.diag(-1*torch.ones(nspools_vr)).to(device)

	fcwdl3 = 1 - fcwdl2
	
	transfer_fraction = [fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, fcwdl3]
	for j in range(nlevdecomp):
		a_ma_vr[(3-1)*nlevdecomp+j,(1-1)*nlevdecomp+j] = transfer_fraction[8]
		a_ma_vr[(4-1)*nlevdecomp+j,(1-1)*nlevdecomp+j] = transfer_fraction[9]
		a_ma_vr[(5-1)*nlevdecomp+j,(2-1)*nlevdecomp+j] = transfer_fraction[0]
		a_ma_vr[(5-1)*nlevdecomp+j,(3-1)*nlevdecomp+j] = transfer_fraction[1]
		a_ma_vr[(5-1)*nlevdecomp+j,(6-1)*nlevdecomp+j] = transfer_fraction[5]
		a_ma_vr[(5-1)*nlevdecomp+j,(7-1)*nlevdecomp+j] = transfer_fraction[7]
		a_ma_vr[(6-1)*nlevdecomp+j,(4-1)*nlevdecomp+j] = transfer_fraction[2]
		a_ma_vr[(6-1)*nlevdecomp+j,(5-1)*nlevdecomp+j] = transfer_fraction[3]
		a_ma_vr[(7-1)*nlevdecomp+j,(5-1)*nlevdecomp+j] = transfer_fraction[4]
		a_ma_vr[(7-1)*nlevdecomp+j,(6-1)*nlevdecomp+j] = transfer_fraction[6]
	# end for ilayer
	return a_ma_vr
# end def a_matrix


# If a_ma is a block matrix where each A_{ij} is (nlevdecomp x nlevdecomp),
# fills in the diagonal of block A_{ij} with "value".
# "value" is assumed to be a tensor with a single element.
# For consistency with the paper (Lu et al. 2020), i and j are indexed from 1.
# Modifies a_ma in place.
def fill_submatrix_diagonal(a_ma, nlevdecomp, i, j, value):
	a_ma[range((i-1)*nlevdecomp, i*nlevdecomp), range((j-1)*nlevdecomp, j*nlevdecomp)] = value  # diag_vector


def a_matrix_vectorized(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector):
	device = fl1s1.device
	nlevdecomp = n_soil_layer
	nspools = npool
	nspools_vr = npool_vr

	# a diagonal matrix
	a_ma_vr = torch.diag(-1*torch.ones(nspools_vr, device=device))

	fcwdl3 = 1 - fcwdl2
	transfer_fraction = [fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, fcwdl3]
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 3, 1, transfer_fraction[8])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 4, 1, transfer_fraction[9])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 2, transfer_fraction[0])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 3, transfer_fraction[1])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 6, transfer_fraction[5])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 5, 7, transfer_fraction[7])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 6, 4, transfer_fraction[2])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 6, 5, transfer_fraction[3])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 7, 5, transfer_fraction[4])
	fill_submatrix_diagonal(a_ma_vr, nlevdecomp, 7, 6, transfer_fraction[6])
	return a_ma_vr


def kk_matrix(xit, xiw, xio, xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3):
	device = xit.device
	
	n_soil_layer = 20
	days_per_year = 365
	# env scalars
	n_scalar = xin[:n_soil_layer] # nitrogen 
	t_scalar = xit[:n_soil_layer] # temperature
	w_scalar = xiw[:n_soil_layer] # water
	o_scalar = xio[:n_soil_layer] # oxygen
	
	# decomposition rate
	kl1 = 1/(days_per_year * tau4l1)
	kl2 = 1/(days_per_year  * tau4l2)
	kl3 = 1/(days_per_year  * tau4l3)
	ks1 = 1/(days_per_year  * tau4s1)
	ks2 = 1/(days_per_year  * tau4s2)
	ks3 = 1/(days_per_year  * tau4s3)
	kcwd = 1/(days_per_year  * tau4cwd)

	decomp_depth_efolding = efolding

	depth_scalar = torch.exp(-zsoi/decomp_depth_efolding)
	xi_tw = t_scalar*w_scalar*o_scalar
	
	kk_ma_vr = (torch.zeros([npool_vr, npool_vr])).to(device)
	for j in range(n_soil_layer):
		kk_ma_vr[0*n_soil_layer+j, 0*n_soil_layer+j] = kcwd * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[1*n_soil_layer+j, 1*n_soil_layer+j] = kl1 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[2*n_soil_layer+j, 2*n_soil_layer+j] = kl2 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[3*n_soil_layer+j, 3*n_soil_layer+j] = kl3 * xi_tw[j] * depth_scalar[j] * n_scalar[j]
		kk_ma_vr[4*n_soil_layer+j, 4*n_soil_layer+j] = ks1 * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[5*n_soil_layer+j, 5*n_soil_layer+j] = ks2 * xi_tw[j] * depth_scalar[j]
		kk_ma_vr[6*n_soil_layer+j, 6*n_soil_layer+j] = ks3 * xi_tw[j] * depth_scalar[j]
	# end for j
	# torch.diag(kk_ma_vr)[0::n_soil_layer] = kcwd * xi_tw * depth_scalar

	return kk_ma_vr

# end def kk_matrix


def kk_matrix_vectorized(xit, xiw, xio, xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3):
	device = xit.device
	
	n_soil_layer = 20
	days_per_year = 365
	# env scalars
	n_scalar = xin[:n_soil_layer] # nitrogen 
	t_scalar = xit[:n_soil_layer] # temperature
	w_scalar = xiw[:n_soil_layer] # water
	o_scalar = xio[:n_soil_layer] # oxygen
	
	# decomposition rate
	kl1 = 1/(days_per_year * tau4l1)
	kl2 = 1/(days_per_year  * tau4l2)
	kl3 = 1/(days_per_year  * tau4l3)
	ks1 = 1/(days_per_year  * tau4s1)
	ks2 = 1/(days_per_year  * tau4s2)
	ks3 = 1/(days_per_year  * tau4s3)
	kcwd = 1/(days_per_year  * tau4cwd)

	decomp_depth_efolding = efolding

	depth_scalar = torch.exp(-zsoi/decomp_depth_efolding)[:n_soil_layer]  # NOTE added the [:n_soil_layer]
	xi_tw = t_scalar*w_scalar*o_scalar

	diagonal_vector = torch.concatenate([kcwd * xi_tw * depth_scalar,
									     kl1 * xi_tw * depth_scalar * n_scalar,
										 kl2 * xi_tw * depth_scalar * n_scalar,
										 kl3 * xi_tw * depth_scalar * n_scalar,
										 ks1 * xi_tw * depth_scalar,
										 ks2 * xi_tw * depth_scalar,
										 ks3 * xi_tw * depth_scalar], dim=0).to(device)
	assert diagonal_vector.shape == (140, )
	return torch.diag(diagonal_vector)
	# return kk_ma_vr

						
# only diffusion
def tri_matrix_alternative(nbedrock, slope, intercept, device):
	# slope = -1.2
	# intercept = -4
	rate_to_atmos = -0. # # at the surface, part of the CO2 should be released to atmos
	transport_rate = -10**(intercept + slope*torch.log10(zsoi[0:20]))
	transport_rate[nbedrock:] = -10**(-30)
	tri_ma_middle = torch.zeros(n_soil_layer, n_soil_layer, device=device)
	for ilayer in range(n_soil_layer):
		if ilayer == 0:
			tri_ma_middle[ilayer, ilayer] = -1*(transport_rate[ilayer] + rate_to_atmos) # at the surface, part of the CO2 should be released to atmos
			tri_ma_middle[ilayer, (ilayer+1)] = transport_rate[ilayer]
		elif ilayer == (n_soil_layer-1):
			tri_ma_middle[ilayer, ilayer] = -1*transport_rate[ilayer]
			tri_ma_middle[ilayer, (ilayer-1)] = transport_rate[ilayer]
		else:
			tri_ma_middle[ilayer, ilayer] = -2*transport_rate[ilayer]
			tri_ma_middle[ilayer, (ilayer-1)] = transport_rate[ilayer]
			tri_ma_middle[ilayer, (ilayer+1)] = transport_rate[ilayer]
	#end for ilayer in range(n_soil_layer):
	tri_ma = torch.zeros([npool_vr, npool_vr], device=device)
	for ipool in range(1, npool):  # Changed so that the first pool (coarse woody debris) is all zeroes
		tri_ma[(ipool*n_soil_layer):((ipool+1)*n_soil_layer), (ipool*n_soil_layer):((ipool+1)*n_soil_layer)] = tri_ma_middle
	# end for ipool in range(npool):
	return tri_ma
# end  def tri_matrix_gas()


def tri_matrix_alternative_vectorized(nbedrock, slope, intercept, intercept_leach, device):
	# Use torch.diag with offset
	# slope = -1.2
	# intercept = -4
	rate_to_atmos = -0. # # at the surface, part of the CO2 should be released to atmos
	transport_rate_float = -10**(intercept + slope*torch.log10(zsoi[0:20]*100)) # convert zsoi from m to cm
	transport_rate_float[nbedrock:] = -10**(-30)
	transport_rate_leach = -10**(intercept_leach + slope*torch.log10(zsoi[0:20]*100)) # convert zsoi from m to cm
	transport_rate_leach[nbedrock:] = -10**(-30)

	# Create a tridiagonal matrix for each pool type
	tri_ma_middle = torch.zeros(n_soil_layer, n_soil_layer, device=device)
	tri_ma_middle = torch.diagonal_scatter(tri_ma_middle, -1*(transport_rate_float[0:n_soil_layer]+transport_rate_leach[0:n_soil_layer]), offset=0)
	tri_ma_middle = torch.diagonal_scatter(tri_ma_middle, transport_rate_float[1:n_soil_layer], offset=1)  # 1 above the main diagonal
	tri_ma_middle = torch.diagonal_scatter(tri_ma_middle, transport_rate_leach[0:(n_soil_layer-1)], offset=-1)  # 1 below the main diagonal
	tri_ma_middle[0, 0] = -1*(transport_rate_leach[0] + rate_to_atmos)
	tri_ma_middle[n_soil_layer-1, n_soil_layer-1] = -1*transport_rate_float[n_soil_layer-1]
	zero_matrix = torch.zeros(n_soil_layer, n_soil_layer, device=device)

	tri_ma = torch.block_diag(zero_matrix, tri_ma_middle, tri_ma_middle, tri_ma_middle,
				              tri_ma_middle, tri_ma_middle, tri_ma_middle)  # First is zero matrix since CWD vertical mixing is not allowed
	
	return tri_ma


# Import the improved code
def tri_matrix_old_improved(nbedrock, altmax, altmax_lastyear, som_diffus, som_adv_flux, cryoturb_diffusion_k):

	device = nbedrock.device

	nlevdecomp = n_soil_layer
	epsilon = 1e-30

	# change the unit from m2/yr to m2/day
	som_diffus_day = som_diffus / days_per_year
	som_adv_flux_day = som_adv_flux / days_per_year # float does not require grad
	cryoturb_diffusion_k_day = cryoturb_diffusion_k / days_per_year
	# print("som_diffus_day.requires_grad: ", som_diffus_day.requires_grad)

	# print("cryoturb_diffusion_k_day.requires_grad: ", cryoturb_diffusion_k_day.requires_grad)

	tri_ma = (torch.zeros([npool_vr, npool_vr])).to(device)

	som_adv_coef = torch.zeros(nlevdecomp+1).to(device) # SOM advective flux (m/day)
	som_diffus_coef = torch.zeros(nlevdecomp+1).to(device) # SOM diffusivity due to bio/cryo-turbation (m2/day)
	diffus = torch.zeros(nlevdecomp+1).to(device) # diffusivity (m2/day)  (includes spinup correction, if any)
	adv_flux = torch.zeros(nlevdecomp+1).to(device) # advective flux (m/day)  (includes spinup correction, if any)

	a_tri_e = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix
	b_tri_e = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix
	c_tri_e = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix
	r_tri_e = torch.zeros(nlevdecomp).to(device) #"r" vector for tridiagonal solution

	a_tri_dz = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix with considering the depth
	b_tri_dz = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix with considering the depth
	c_tri_dz = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix with considering the depth

	# d_p1_zp1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for next j
	# # (set to zero for no diffusion)
	# d_m1_zm1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for previous j
	# # (set to zero for no diffusion)
	f_p1 = torch.zeros(nlevdecomp+1).to(device) # water flux for next j
	f_m1 = torch.zeros(nlevdecomp+1).to(device) # water flux for previous j
	pe_p1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for next j
	pe_m1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for previous j
	
	w_p1 = torch.zeros(nlevdecomp+1).to(device)
	w_m1 = torch.zeros(nlevdecomp+1).to(device)

	#------ first get diffusivity / advection terms -------
	# Convert conditions to tensor operations
	active_layer_depth = torch.tensor(max(altmax.item(), altmax_lastyear.item())).to(device)
	# is_active_layer = zisoi[:nbedrock+1] < active_layer_depth
	# is_below_active_layer_and_cryoturb = (zisoi[:nbedrock+1] >= active_layer_depth) & (zisoi[:nbedrock+1] <= torch.min(torch.tensor(max_depth_cryoturb), zisoi[nbedrock+1]))
	is_active_layer = zisoi[:nlevdecomp+1] < active_layer_depth
	is_below_active_layer_and_cryoturb = (zisoi[:nlevdecomp+1] >= active_layer_depth) & (zisoi[:nlevdecomp+1] <= torch.min(torch.tensor(max_depth_cryoturb), zisoi[nlevdecomp+1]))
	is_bedrock_layer = torch.arange(nlevdecomp+1).to(device) > nbedrock
	# Fill is_active_layer with False for bedrock layers to shape = nlevdecomp+1
	# is_active_layer = torch.cat((is_active_layer, torch.full((nlevdecomp-nbedrock,), False).to(device)))
	# is_below_active_layer_and_cryoturb = torch.cat((is_below_active_layer_and_cryoturb, torch.full((nlevdecomp-nbedrock,), False).to(device)))
	# is_bedrock_layer = torch.cat((is_bedrock_layer, torch.full((nlevdecomp-nbedrock,), True).to(device)))

	# Initialize coefficients with zeros
	som_diffus_coef.fill_(0.)
	som_adv_coef.fill_(0.)

	if active_layer_depth <= max_altdepth_cryoturbation and active_layer_depth > 0.:
		# Active layer conditions
		# print("nbedrock: ", nbedrock)
		# print("Shape of som_diffus_coef: ", som_diffus_coef.shape)
		# print("Shape of active_layer_depth: ", active_layer_depth.shape)
		# print("Shape of zisoi[:nbedrock+1]: ", zisoi[:nbedrock+1].shape)
		# print("Shape of is_active_layer: ", is_active_layer.shape)
		# print("is_active_layer: ", is_active_layer)
		# print("Shape of is_below_active_layer_and_cryoturb: ", is_below_active_layer_and_cryoturb.shape)
		# print("is_below_active_layer_and_cryoturb: ", is_below_active_layer_and_cryoturb)
		# print("Shape of is_bedrock_layer: ", is_bedrock_layer.shape)
		# print("is_bedrock_layer: ", is_bedrock_layer)
		# print("Shape of cryoturb_diffusion_k_day: ", cryoturb_diffusion_k_day.shape)
		som_diffus_coef[is_active_layer] = cryoturb_diffusion_k_day
		linear_decrease_factor = (1. - (zisoi[:nlevdecomp+1][is_below_active_layer_and_cryoturb] - active_layer_depth) / (torch.min(torch.tensor(max_depth_cryoturb), zisoi[nlevdecomp+1]) - active_layer_depth))
		# print("Shape of linear_decrease_factor: ", linear_decrease_factor.shape)
		# print("Shape of som_diffus_coef: ", som_diffus_coef.shape)
		# print("Shape of is_below_active_layer_and_cryoturb: ", is_below_active_layer_and_cryoturb.shape)
		# print(torch.maximum(cryoturb_diffusion_k * linear_decrease_factor, torch.tensor(0.)).shape)
		# print("Shape of som_diffus_coef[is_below_active_layer_and_cryoturb]: ", som_diffus_coef[is_below_active_layer_and_cryoturb].shape)
		som_diffus_coef[is_below_active_layer_and_cryoturb] = torch.maximum(cryoturb_diffusion_k_day * linear_decrease_factor, torch.tensor(0.))
	elif active_layer_depth > 0.:
		# Constant advection and diffusion up to bedrock
		som_adv_coef[:nbedrock+1] = som_adv_flux_day
		som_diffus_coef[:nbedrock+1] = som_diffus_day
	# No else clause needed for completely frozen soils since the initialization already sets coefficients to 0

	# Apply mask for bedrock layers (no advection or diffusion)
	som_adv_coef[is_bedrock_layer] = 0.
	som_diffus_coef[is_bedrock_layer] = 0.


	# Initial setup - vectorized
	adv_flux = torch.where(torch.abs(som_adv_coef) < epsilon, torch.full_like(som_adv_coef, epsilon), som_adv_coef)
	diffus = torch.where(torch.abs(som_diffus_coef) < epsilon, torch.full_like(som_diffus_coef, epsilon), som_diffus_coef)
	# print("diffus", diffus)
	# print("adv_flux requires grad", adv_flux.requires_grad)
	# print("diffus requires grad", diffus.requires_grad)

	# Initialize tensors
	f_m1 = adv_flux.clone()
	f_p1 = torch.cat((adv_flux[1:], torch.tensor([0.]).to(device)))  # Shift adv_flux down and pad with 0
	w_m1 = torch.zeros_like(adv_flux)
	w_p1 = torch.zeros_like(adv_flux)
	d_m1_zm1 = torch.zeros_like(adv_flux)
	d_p1_zp1 = torch.zeros_like(adv_flux)

	# Calculations that apply for all layers except the special cases at the top and bottom
	# w_m1[1:] = (zisoi[:nlevdecomp+1][:-1] - zsoi[:nlevdecomp+1][:-1]) / dz_node[:nlevdecomp+1][1:]  # Skip the first layer for w_m1
	# Try to avoid inplace operations
	w_m1 = torch.cat([w_m1[:1], (zisoi[:nlevdecomp+1][:-1] - zsoi[:nlevdecomp+1][:-1]) / dz_node[:nlevdecomp+1][1:]])
	# At the bottom, assume no gradient in dz (i.e., they're the same)
	# w_p1[:-2] = (zsoi[:nlevdecomp+1][1:-1] - zisoi[:nlevdecomp+1][:-2]) / dz_node[:nlevdecomp+1][1:-1]  # Skip the last two layers for w_p1
	# Try to avoid inplace operations
	w_p1 = torch.cat([(zsoi[:nlevdecomp+1][1:-1] - zisoi[:nlevdecomp+1][:-2]) / dz_node[:nlevdecomp+1][1:-1], w_p1[-2:]])
	

	# Harmonic mean for internal layers, with adjustments for boundary conditions
	inner_layers = (diffus[1:] > 0) & (diffus[:-1] > 0)  # Boolean mask for layers where both adjacent diffusivities are > 0
	# d_m1_zm1[1:] = torch.where(inner_layers, 1. / ((1. - w_m1[1:]) / diffus[1:] + w_m1[1:] / diffus[:-1]), 0.)
	# d_p1_zp1[:-1] = torch.where(inner_layers, 1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]), (1. - w_m1[:-1]) * diffus[:-1] + w_p1[:-1] * diffus[1:])
	d_m1_zm1 = torch.cat([d_m1_zm1[:1], torch.where(inner_layers, 1. / ((1. - w_m1[1:]) / diffus[1:] + w_m1[1:] / diffus[:-1]), torch.zeros_like(d_m1_zm1[1:]))])

	# Original line:
	# d_p1_zp1_temp = torch.where(inner_layers, 1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]), (1. - w_m1[:-1]) * diffus[:-1] + w_p1[:-1] * diffus[1:])

	# Fixed line:
	d_p1_zp1_temp = torch.where(inner_layers, 1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]), (1. - w_p1[:-1]) * diffus[:-1] + w_p1[:-1] * diffus[1:])  # NOTE: Replaced 1-w_m1 with 1-w_p1, I believe this was a typo in the original Fortran code.
	d_p1_zp1 = torch.cat([d_p1_zp1_temp, d_p1_zp1[-1:]])

	# Adjust for dz_node scaling
	d_m1_zm1[1:] /= dz_node[:nlevdecomp+1][1:]
	d_p1_zp1[:-1] /= dz_node[:nlevdecomp+1][1:]

	# print("d_p1_zp1", d_p1_zp1)

	# Layer lower than nbedrock-1: d_p1_zp1 = d_m1_zm1
	

	# Bottom layer - assume no gradient in dz
	# w_m1[-1] = (zisoi[:nlevdecomp+1][-2] - zsoi[:nlevdecomp+1][-2]) / dz_node[:nlevdecomp+1][-1]
	# d_m1_zm1[-1] = 1. / ((1. - w_m1[-1]) / diffus[-1] + w_m1[-1] / diffus[-2]) if diffus[-1] > 0 and diffus[-2] > 0 else 0.
	# d_m1_zm1[-1] /= dz_node[:nlevdecomp+1][-1]
	w_m1_bottom = (zisoi[:nlevdecomp+1][-2] - zsoi[:nlevdecomp+1][-2]) / dz_node[:nlevdecomp+1][-1]
	d_m1_zm1_bottom = 1. / ((1. - w_m1_bottom) / diffus[-1] + w_m1_bottom / diffus[-2]) if diffus[-1] > 0 and diffus[-2] > 0 else 0.
	d_m1_zm1_bottom /= dz_node[:nlevdecomp+1][-1]

	d_m1_zm1= torch.cat([d_m1_zm1[:-1], d_m1_zm1_bottom.unsqueeze(0)])
	d_p1_zp1 = torch.cat([d_p1_zp1[:nbedrock], d_m1_zm1[nbedrock:]])

	# d_p1_zp1[nbedrock-1:] = d_m1_zm1[nbedrock-1:]
	# # No advective flux for the layer between nbedrock and nlevdecomp
	# f_p1[nbedrock-1:] = 0

	# No advective flux for the layer between nbedrock and nlevdecomp without in-place modification
	f_p1 = torch.cat([f_p1[:nbedrock], torch.zeros_like(f_p1[nbedrock:])])

	# assign 0 if nlevdecomp > nbedrock
	# for example, if nbedrock = 7, only select top 6 elements
	w_p1 = torch.where(torch.arange(nlevdecomp+1).to(device) > nbedrock-1, torch.zeros_like(w_p1), w_p1)

	# Boundary conditions
	# Top layer
	w_p1[0] = (zsoi[:nlevdecomp+1][1] - zisoi[:nlevdecomp+1][0]) / dz_node[:nlevdecomp+1][1]
	d_p1_zp1[0] = 1. / ((1. - w_p1[0]) / diffus[0] + w_p1[0] / diffus[1]) if diffus[1] > 0 and diffus[0] > 0 else 0.
	d_p1_zp1[0] /= dz_node[:nlevdecomp+1][1]

	# print((zsoi[:nlevdecomp+1][1] - zisoi[:nlevdecomp+1][0]) / dz_node[:nlevdecomp+1][1])

	# print("w_p1", w_p1)
	# print("w_m1", w_m1)
	# # print(1. / ((1. - w_p1[:-1]) / diffus[:-1] + w_p1[:-1] / diffus[1:]))
	# print("d_p1_zp1", d_p1_zp1)
	# print("d_m1_zm1", d_m1_zm1)




	# Peclet numbers
	pe_m1 = torch.where(d_m1_zm1 == 0, torch.zeros_like(f_m1), f_m1 / (d_m1_zm1))  # NOTE + epsilon))
	pe_p1 = torch.where(d_p1_zp1 == 0, torch.zeros_like(f_p1), f_p1 / (d_p1_zp1))  # NOTE + epsilon))

	# Pre-compute the 'aaa' values for Patankar functions
	aaa_m = torch.maximum(torch.zeros_like(pe_m1), (1. - 0.1 * pe_m1.abs())**5)
	aaa_p = torch.maximum(torch.zeros_like(pe_p1), (1. - 0.1 * pe_p1.abs())**5)

	# Vectorized computation of tridiagonal coefficients
	a_tri_e = -(d_m1_zm1 * aaa_m + torch.maximum(f_m1, torch.zeros_like(f_m1)))
	c_tri_e = -(d_p1_zp1 * aaa_p + torch.maximum(-f_p1, torch.zeros_like(f_p1)))
	b_tri_e = -a_tri_e - c_tri_e

	# print("aaa_p", aaa_p)
	# print("f_p1", f_p1)

	# print("Shape of d_m1_zm1: ", d_m1_zm1.shape)
	# print("Shape of pe_m1: ", pe_m1.shape)
	# print("Shape of pe_p1: ", pe_p1.shape)
	# print("Shape of aaa_m: ", aaa_m.shape)
	# print("Shape of f_m1: ", f_m1.shape)



	a_tri_dz = a_tri_e / dz[:nlevdecomp+1]
	b_tri_dz = b_tri_e / dz[:nlevdecomp+1]
	c_tri_dz = c_tri_e / dz[:nlevdecomp+1]
	a_tri_dz = a_tri_dz[:-1]
	b_tri_dz = b_tri_dz[:-1]
	c_tri_dz = c_tri_dz[:-1]

	# # no vertical transportation in CWD
	# for i in range(1, npool): #= 2 : npool
	# 	for j in range(nlevdecomp): #= 1 : nlevdecomp
	# 		tri_ma[j+(i)*nlevdecomp,j+(i)*nlevdecomp] = b_tri_dz[j]
	# 		if j == 0:   # upper boundary
	# 			tri_ma[j+(i)*nlevdecomp,j+(i)*nlevdecomp] = -c_tri_dz[j]
	# 			# print(i, j, 1+(i)*nlevdecomp, tri_ma[1+(i)*nlevdecomp,1+(i)*nlevdecomp])

	# 		# end if j == 0: 
	# 		if j == (nlevdecomp-1):  # bottom boundary
	# 			tri_ma[nlevdecomp-1+(i)*nlevdecomp,nlevdecomp-1+(i)*nlevdecomp] = -a_tri_dz[nlevdecomp-1]
	# 		# end if j == (nlevdecomp-1):
	# 		if j < (nlevdecomp-1): # avoid tranfer from for example, litr3_20th layer to soil1_1st layer
	# 			tri_ma[j+(i)*nlevdecomp,j+1+(i)*nlevdecomp] = c_tri_dz[j]

	# 		# end if j < (nlevdecomp-1): 

		
	# 		if j > 0: # avoid tranfer from for example,soil1_1st layer to litr3_20th layer
	# 			tri_ma[j+(i)*nlevdecomp,j-1+(i)*nlevdecomp] = a_tri_dz[j]

	# for i in range(1, npool):
	# 	idx = torch.arange(nlevdecomp) + i * nlevdecomp
	# 	tri_ma[idx, idx] = b_tri_dz

	# 	# Upper boundary
	# 	tri_ma[idx[0], idx[0]] = -c_tri_dz[0]

	# 	# Bottom boundary
	# 	tri_ma[idx[-1], idx[-1]] = -a_tri_dz[-1]

	# 	# Off-diagonals
	# 	idx = torch.arange(nlevdecomp-1) + i * nlevdecomp
	# 	tri_ma[idx, idx+1] = c_tri_dz[:-1]
	# 	tri_ma[idx+1, idx] = a_tri_dz[1:]

	# Try to get rid of the for loop
	# Expand a_tri_dz, b_tri_dz, c_tri_dz
	expanded_a = torch.cat([torch.zeros(20, device=device), a_tri_dz.repeat(npool-1)])
	expanded_b = torch.cat([torch.zeros(20, device=device), b_tri_dz.repeat(npool-1)])
	expanded_c = torch.cat([torch.zeros(20, device=device), c_tri_dz.repeat(npool-1)])

	# Fill diagonal with expanded_b
	tri_ma.fill_diagonal_(0)  # Ensure diagonal is clear before setting
	tri_ma[torch.arange(140), torch.arange(140)] = expanded_b

	# Adjust for upper boundary conditions across blocks
	upper_boundary_indices = torch.arange(20, 140, 20)
	tri_ma[upper_boundary_indices, upper_boundary_indices] = -expanded_c[upper_boundary_indices]

	# Adjust for bottom boundary conditions across blocks
	bottom_boundary_indices = torch.arange(39, 140, 20)
	tri_ma[bottom_boundary_indices, bottom_boundary_indices] = -expanded_a[bottom_boundary_indices]

	# Fill right off-diagonal with c_tri_dz
	# off_diag_indices_right = torch.arange(19, 139)  # Right off-diagonal
	# tri_ma[torch.arange(19, 139), torch.arange(20, 140)] = expanded_c[off_diag_indices_right]
	tri_ma[torch.arange(20, 139), torch.arange(21, 140)] = expanded_c[torch.arange(20, 139)]  # NOTE changed

	# Fill left off-diagonal with a_tri_dz
	# off_diag_indices_left = torch.arange(20, 140)  # Left off-diagonal
	# tri_ma[torch.arange(20, 140), torch.arange(19, 139)] = expanded_a[off_diag_indices_left]
	tri_ma[torch.arange(21, 140), torch.arange(20, 139)] = expanded_a[torch.arange(21, 140)]  # NOTE changed

	# Adjust for upper boundary conditions across blocks
	tri_ma[upper_boundary_indices, upper_boundary_indices-1] = 0
	
	# Adjust for bottom boundary conditions across blocks
	bottom_boundary_indices = torch.arange(39, 120, 20)
	tri_ma[bottom_boundary_indices, bottom_boundary_indices+1] = 0




	# # no vertical transportation in CWD
	# for i in range(1, npool): #= 2 : npool
	# 	for j in range(nlevdecomp): #= 1 : nlevdecomp
	# 		tri_ma[j+(i)*nlevdecomp,j+(i)*nlevdecomp] = b_tri_dz[j]
	# 		if j == 0:   # upper boundary
	# 			tri_ma[1+(i)*nlevdecomp,1+(i)*nlevdecomp] = -c_tri_dz[1]
	# 			# print(i, j, 1+(i)*nlevdecomp, tri_ma[1+(i)*nlevdecomp,1+(i)*nlevdecomp])

	# 		# end if j == 0: 
	# 		if j == (nlevdecomp-1):  # bottom boundary
	# 			tri_ma[nlevdecomp-1+(i)*nlevdecomp,nlevdecomp-1+(i)*nlevdecomp] = -a_tri_dz[nlevdecomp-1]
	# 		# end if j == (nlevdecomp-1):
	# 		if j < (nlevdecomp-1): # avoid tranfer from for example, litr3_20th layer to soil1_1st layer
	# 			tri_ma[j+(i)*nlevdecomp,j+1+(i)*nlevdecomp] = c_tri_dz[j]

	# 		# end if j < (nlevdecomp-1): 

		
	# 		if j > 0: # avoid tranfer from for example,soil1_1st layer to litr3_20th layer
	# 			tri_ma[j+(i)*nlevdecomp,j-1+(i)*nlevdecomp] = a_tri_dz[j]

	

	return tri_ma


def tri_matrix_old(nbedrock, altmax, altmax_lastyear, som_diffus, som_adv_flux, cryoturb_diffusion_k):
	device = nbedrock.device

	nlevdecomp = n_soil_layer
	epsilon = 1e-30

	# change the unit from m2/yr to m2/day, in consistant with the unit of input
	som_diffus = som_diffus/days_per_year
	som_adv_flux = som_adv_flux/days_per_year
	cryoturb_diffusion_k = cryoturb_diffusion_k/days_per_year

	tri_ma = (torch.zeros([npool_vr, npool_vr])).to(device)

	som_adv_coef = torch.zeros(nlevdecomp+1).to(device) # SOM advective flux (m/day)
	som_diffus_coef = torch.zeros(nlevdecomp+1).to(device) # SOM diffusivity due to bio/cryo-turbation (m2/day)
	diffus = torch.zeros(nlevdecomp+1).to(device) # diffusivity (m2/day)  (includes spinup correction, if any)
	adv_flux = torch.zeros(nlevdecomp+1).to(device) # advective flux (m/day)  (includes spinup correction, if any)

	a_tri_e = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix
	b_tri_e = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix
	c_tri_e = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix
	r_tri_e = torch.zeros(nlevdecomp).to(device) #"r" vector for tridiagonal solution

	a_tri_dz = torch.zeros(nlevdecomp).to(device) # "a" vector for tridiagonal matrix with considering the depth
	b_tri_dz = torch.zeros(nlevdecomp).to(device) # "b" vector for tridiagonal matrix with considering the depth
	c_tri_dz = torch.zeros(nlevdecomp).to(device) # "c" vector for tridiagonal matrix with considering the depth

	d_p1_zp1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for next j
	# (set to zero for no diffusion)
	d_m1_zm1 = torch.zeros(nlevdecomp+1).to(device) # diffusivity/delta_z for previous j
	# (set to zero for no diffusion)
	f_p1 = torch.zeros(nlevdecomp+1).to(device) # water flux for next j
	f_m1 = torch.zeros(nlevdecomp+1).to(device) # water flux for previous j
	pe_p1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for next j
	pe_m1 = torch.zeros(nlevdecomp+1).to(device) # Peclet # for previous j
	
	w_p1 = torch.zeros(nlevdecomp+1).to(device)
	w_m1 = torch.zeros(nlevdecomp+1).to(device)
	#------ first get diffusivity / advection terms -------
	# use different mixing rates for bioturbation and cryoturbation, with fixed bioturbation and cryoturbation set to a maximum depth
	if  max(altmax, altmax_lastyear) <= max_altdepth_cryoturbation and max(altmax, altmax_lastyear) > 0.:
		# use mixing profile modified slightly from Koven et al. (2009): constant through active layer, linear decrease from base of active layer to zero at a fixed depth
		for j in range(nlevdecomp+1):
			if j <= nbedrock:
				if zisoi[j] < max(altmax, altmax_lastyear):
					som_diffus_coef[j] = cryoturb_diffusion_k
					som_adv_coef[j] = 0.
				else:
					som_diffus_coef[j] = max(cryoturb_diffusion_k*(1.-(zisoi[j]-max(altmax, altmax_lastyear))/(min(max_depth_cryoturb, zisoi[nbedrock+1]) - max(altmax, altmax_lastyear))), 0.) # go linearly to zero between ALT and max_depth_cryoturb
					som_adv_coef[j] = 0.
				# end if zisoi[j] < max(altmax, altmax_lastyear):
			else:
				som_adv_coef[j] = 0.
				som_diffus_coef[j] = 0.
			# end if j <= nbedrock:
		# end for j in range(nlevdecomp+1):
	elif max(altmax, altmax_lastyear) > 0.:
		# constant advection, constant diffusion
		for j in range(nlevdecomp+1):
			if j <= nbedrock:
				som_adv_coef[j] = som_adv_flux
				som_diffus_coef[j] = som_diffus
			else:
				som_adv_coef[j] = 0.
				som_diffus_coef[j] = 0.
			# end if j <= nbedrock:
		# end for j in range(nlevdecomp+1):
	else:
		# completely frozen soils--no mixing
		for j in range(nlevdecomp+1):
			som_adv_coef[j] = 0.
			som_diffus_coef[j] = 0.
		# end for j in range(nlevdecomp+1):
	# end if  max(altmax, altmax_lastyear) <= max_altdepth_cryoturbation and max(altmax, altmax_lastyear) > 0.:

	# for j in torch.arange(nlevdecomp+1):
	#	if abs(som_adv_coef[j])  < epsilon:
	#		adv_flux[j] = epsilon
	#	else:
	#		adv_flux[j] = som_adv_coef[j]
	#	# end if abs(som_adv_coef[j])  < epsilon:
	#	if abs(som_diffus_coef[j])  < epsilon:
	#		diffus[j] = epsilon
	#	else:
	#		diffus[j] = som_diffus_coef[j]
	#	# end abs(som_diffus_coef[j])  < epsilon:
	## end for j in range(nlevdecomp+1):
	
	# Set Pe (Peclet #) and D/dz throughout column
	for j in range(nlevdecomp+1):
		if abs(som_adv_coef[j])  < epsilon:
			adv_flux[j] = epsilon
		else:
			adv_flux[j] = som_adv_coef[j]
		# end if abs(som_adv_coef[j])  < epsilon:
		
		if abs(som_diffus_coef[j])  < epsilon:
			diffus[j] = epsilon
		else:
			diffus[j] = som_diffus_coef[j]
		# end if abs(som_diffus_coef[j])  < epsilon:

	for j in range(nlevdecomp+1):  # NOTE changed so all diffus get calculated above
		# Calculate the D and F terms in the Patankar algorithm
		if j == 0:
			d_m1_zm1[j] = 0.
			w_m1[j] = 0.
			w_p1[j] = (zsoi[j+1] - zisoi[j]) / dz_node[j+1]
			
			if diffus[j+1] > 0. and diffus[j] > 0.:
				d_p1_zp1[j] = 1. / ((1. - w_p1[j].clone()) / diffus[j].clone() + w_p1[j].clone() / diffus[j+1].clone()) # Harmonic mean of diffus
			else:
				d_p1_zp1[j] = 0.
			# end diffus[j+1] > 0. and diffus[j] > 0.:
			
			d_p1_zp1[j] = d_p1_zp1[j] / dz_node[j+1]
			f_m1[j] = adv_flux[j]  # Include infiltration here
			f_p1[j] = adv_flux[j+1]

			# pe_m1[j] = 0.
			# pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j] # Peclet #
		elif j >= nbedrock:  # NOTE used to be nbedrock-1:
			# At the bottom, assume no gradient in d_z (i.e., they're the same)
			w_m1[j] = (zisoi[j-1] - zsoi[j-1]) / dz_node[j]
			w_p1[j] = 0.

			if diffus[j] > 0. and diffus[j-1] > 0.:
				d_m1_zm1[j] = 1. / ((1. - w_m1[j].clone()) / diffus[j].clone() + w_m1[j].clone() / diffus[j-1].clone()) # Harmonic mean of diffus
			else:
				d_m1_zm1[j] = 0.
			# end diffus[j] > 0. and diffus[j-1] > 0.:
			
			d_m1_zm1[j] = d_m1_zm1[j] / dz_node[j]
			d_p1_zp1[j] = d_m1_zm1[j] # Set to be the same
			f_m1[j] = adv_flux[j]
			# f_p1(j) = adv_flux(j+1)
			f_p1[j] = 0.

			# pe_m1[j] = f_m1[j].clone() / d_m1_zm1[j] # Peclet #
			# pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j] # Peclet #
		else:
			# Use distance from j-1 node to interface with j divided by distance between nodes
			w_m1[j] = (zisoi[j-1] - zsoi[j-1]) / dz_node[j]
			w_p1[j] = (zsoi[j+1] - zisoi[j]) / dz_node[j+1]
			
			if diffus[j-1] > 0. and diffus[j] > 0.:
				d_m1_zm1[j] = 1. / ((1. - w_m1[j].clone()) / diffus[j].clone() + w_m1[j].clone() / diffus[j-1].clone()) # Harmonic mean of diffus
			else:
				d_m1_zm1[j] = 0.
			# end diffus[j-1] > 0. and diffus[j] > 0.:

			if diffus[j+1] > 0. and diffus[j] > 0.:
				d_p1_zp1[j] = 1. / ((1. - w_p1[j].clone()) / diffus[j].clone() + w_p1[j].clone() / diffus[j+1].clone()) # Harmonic mean of diffus
			else:
				d_p1_zp1[j] = (1. - w_p1[j].clone()) * diffus[j].clone() + w_p1[j].clone() * diffus[j+1].clone() # Arithmetic mean of diffus.  NOTE: Replaced 1-w_m1 with 1-w_p1, I believe this was a typo in the original Fortran code.
			# end if diffus[j+1] > 0. and diffus[j] > 0.:
			
			d_m1_zm1[j] = d_m1_zm1[j] / dz_node[j]
			d_p1_zp1[j] = d_p1_zp1[j] / dz_node[j+1]
			f_m1[j] = adv_flux[j]
			f_p1[j] = adv_flux[j+1]
			# pe_m1[j] = f_m1[j].clone() / d_m1_zm1[j] # Peclet #
			# pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j] # Peclet #
		# end j == 0:
	# end for j in range(nlevdecomp+1): 

	# Peclet #
	for j in range(nlevdecomp+1):
		if d_m1_zm1[j] == 0.:
			pe_m1[j] = 0.
		else:
			pe_m1[j] = f_m1[j].clone() / d_m1_zm1[j]
		#end if d_m1_zm1[j] == 0.:
		if d_p1_zp1[j] == 0.:
			pe_p1[j] = 0.
		else:
			pe_p1[j] = f_p1[j].clone() / d_p1_zp1[j]
	# end for j in torch.arange(nlevdecomp+1):

	# Calculate the tridiagonal coefficients
	for j in range(-1, (nlevdecomp+1)): # 0:nlevdecomp+1
		if j == -1: # top layer (atmosphere)
			pass
			# a_tri(j) = 0.
			# b_tri(j) = 1.
			# c_tri(j) = -1.
			# b_tri_e(j) = b_tri(j)
		elif j == 0: #  Set statement functions
			aaa = max (0., (1. - 0.1 * abs(pe_m1[j]))**5)  # A function from Patankar, Table 5.2, pg 95
			a_tri_e[j] = -(d_m1_zm1[j] * aaa + max( f_m1[j], 0.)) # Eqn 5.47 Patankar
			aaa = max (0., (1. - 0.1 * abs(pe_p1[j]))**5)  # A function from Patankar, Table 5.2, pg 95
			c_tri_e[j] = -(d_p1_zp1[j] * aaa + max(-f_p1[j], 0.))
			b_tri_e[j] = -a_tri_e[j] - c_tri_e[j]
		elif j < nlevdecomp:
			aaa = max (0., (1. - 0.1 * abs(pe_m1[j]))**5) # A function from Patankar, Table 5.2, pg 95
			a_tri_e[j] = -(d_m1_zm1[j] * aaa + max( f_m1[j], 0.)) # Eqn 5.47 Patankar
			aaa = max (0., (1. - 0.1 * abs(pe_p1[j]))**5)  # A function from Patankar, Table 5.2, pg 95
			c_tri_e[j] = -(d_p1_zp1[j] * aaa + max(-f_p1[j], 0.))
			b_tri_e[j] = -a_tri_e[j] - c_tri_e[j]
		else: # j==nlevdecomp; 0 concentration gradient at bottom
			pass
			# a_tri(j) = -1.
			# b_tri(j) = 1.
			# c_tri(j) = 0.
		# end if j == 0: 
	# end for j in range(nlevdecomp+2)
	
	for j in range(nlevdecomp):
		# elements for vertical matrix match unit g/m3,  Tri/dz
		a_tri_dz[j] = a_tri_e[j] / dz[j]
		b_tri_dz[j] = b_tri_e[j] / dz[j]
		c_tri_dz[j] = c_tri_e[j] / dz[j]
	# end for j in range(nlevdecomp):


	for i in range(1, npool):
		start_idx = i * nlevdecomp
		end_idx = (i + 1) * nlevdecomp

		# Main diagonal
		diag_indices = torch.arange(start_idx, end_idx)
		tri_ma[diag_indices, diag_indices] = b_tri_dz

		# Upper boundary condition adjustment
		if nlevdecomp > 1:
			tri_ma[start_idx, start_idx] = -c_tri_dz[0]   # NOTE used to be -c_tri_dz[1]


		# Bottom boundary condition adjustment
		tri_ma[end_idx - 1, end_idx - 1] = -a_tri_dz[-1]

		# Upper and lower diagonals
		# Ensure indices don't go out of bounds
		if nlevdecomp > 2:  # Only proceed if there are at least 3 layers, allowing for upper and lower diagonals
			upper_diag_indices = diag_indices[:-1] + 1
			lower_diag_indices = diag_indices[1:] - 1

			tri_ma[diag_indices[:-1], upper_diag_indices] = c_tri_dz[:-1]
			tri_ma[diag_indices[1:], lower_diag_indices] = a_tri_dz[1:]

	return tri_ma

# end def tri_matrix

def catanf(t1):
	catanf_results = 11.75 +(29.7 / np.pi) * torch.arctan( torch.tensor(np.pi) * 0.031  * ( t1 - 15.4 ))
	return catanf_results
# end def catanf
