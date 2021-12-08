import numpy as np
import pyHiChi as hc
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import hichi_primitives as hp

class BiPole:
    """ This is a class that contains the Pythonic backend tools that communicate via the pyHiChi package.
    The class itself is specialized to field geometries of the dipole and bi-dipole waves and the interaction
    with uniform electron beams.
    """

    def __init__(self, wavelength, R0, P0, a_fac, L_box, number_e_real, number_e, electron_energy, beam_length, spot_radius, dipole_dir='y', thresh=1/4, tstart=0.0):
        self.wavelength = wavelength
        self.R0 = R0*self.wavelength
        self.P0 = P0
        self.w = (2*np.pi*hc.c)/self.wavelength
        self.a_fac = a_fac
        self.a=a_fac*self.w
        self.L_box = L_box*self.wavelength
        self.number_e_real = number_e_real
        self.number_e = number_e
        self.electron_energy = electron_energy
        self.beam_length = beam_length*self.wavelength
        self.spot_radius = spot_radius*self.wavelength
        self.distance_start = self.R0
        self.d_sim = 2*self.R0
        self.dipole_dir = dipole_dir
        self.thresh = thresh
        self.chi_max = 100
        self.c = hc.c
        self.me = hc.ELECTRON_MASS
        self.d0 = (self.wavelength**2/(4*np.pi**2))*np.sqrt(3*P0/self.c)
        self.t_aux = self.wavelength/self.c
        self.tstart = tstart
        self.dt_revert = (-self.d_sim)/hc.c
        self.lx, self.ly, self.lz = self.L_box
        self.min_coords = hc.Vector3d(self.lx[0], self.ly[0], self.lz[0])
        self.max_coords = hc.Vector3d(self.lx[1], self.ly[1], self.lz[1])
        self.nx, self.ny, self.nz = ((self.max_coords.x-self.min_coords.x)/(self.thresh*self.wavelength), (self.max_coords.y-self.min_coords.y)/(self.thresh*self.wavelength), (self.max_coords.z-self.min_coords.z)/(self.thresh*self.wavelength))
        self.grid_size = hc.Vector3d(self.nx, self.ny, self.nz)
        self.grid_step = (self.max_coords-self.min_coords)/self.grid_size
        self.weight = self.number_e_real/self.number_e
        self.n_iter = int(self.d_sim/(thresh*self.wavelength))
        self.time_fine_step = self.d_sim/(self.n_iter*self.c)
        self.qed = hc.QED()

        def g(R):
            """[Quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            T = tstart+(R-self.R0)/self.c
            return np.exp(-self.a**2 * T**2)*np.sin(self.w*T)
        self.g = g

        def gp(R):
            """[first derivative of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the derivative of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            T = tstart+(R-self.R0)/self.c
            return np.exp(-self.a**2 * T**2)*(self.w*np.cos(self.w*T)-2*self.a**2*T*np.sin(self.w*T))
        self.gp = gp

        def gpp(R):
            """[second derivative of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the second derivative of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            T = tstart+(R-self.R0)/self.c
            return np.exp(-self.a**2 * T**2)*(np.sin(self.w*T)*(4*self.a**4*T**2 -self.w**2 -2*self.a**2)-4*self.a**2*self.w*T*np.cos(self.w*T))
        self.gpp = gpp

        def gpp_plus(R):
            """[inversed argument sum of the second derivative of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the argument-inversed sum of the second derivative of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            return gpp(-R)+gpp(R)
        self.gpp_plus = gpp_plus

        def gp_minus(R):
            """[inversed argument difference of the first derivative of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the argument-inversed difference of the first derivative of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            return gp(-R)-gp(R)
        self.gp_minus = gp_minus

        def gpp_minus(R):
            """[inversed argument difference of the second derivative of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the argument-inversed difference of the second derivative of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            return gpp(-R)-gpp(R)
        self.gpp_minus = gpp_minus
        
        def gp_plus(R):
            """[inversed argument sum of the first derivative of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the argument-inversed sum of the first derivative of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            return gp(-R)+gp(R)
        self.gp_plus = gp_plus

        def g_minus(R):
            """[inversed argument difference of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the argument-inversed difference of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            return g(-R)-g(R)
        self.g_minus = g_minus

        def g_plus(R):
            """[inversed argument sum of the quasi-Gaussian driving function of a dipole wave.]

            :param R: distance to a coordinate in (x,y,z) space.
            :type R: float
            :return: value of the argument-inversed sum of the quasi-Gaussian driving function at location R
            :rtype: float
            """
            return g(-R)+g(R)
        self.g_plus = g_plus

        def E_dipole(x, y, z):
            """[Returns a pyHiChi electric and magnetic field object of electric dipole geometry at coordinates (x,y,z) in space.]

            :param x,y,z: coordinates in the simulation domain
            :type x,y,z: float
            :return: electric and magnetic pyHiChi field objects at coordinates (x,y,z) of electric dipole geometry
            :rtype: pyHiChi field object
            """
            R = np.sqrt(x**2+y**2+z**2)
            if (R>1e-5):
                ehat = hc.Vector3d(0, 1, 0) #Y-dir
                n_hat = hc.Vector3d(x, y, z)/R
                h1 = -hc.cross(n_hat, self.d0*ehat)
                e1 = hc.cross(h1, n_hat)
                e2 = 3*hc.dot(n_hat, self.d0*ehat)*n_hat - self.d0*ehat

                H_e = h1*(gpp_plus(R)/(self.c*self.c*(R)) + gp_minus(R)/(self.c*(R)**2))
                E_e = e1*(gpp_minus(R)/((R)*self.c**2))+e2*(gp_plus(R)/(self.c*(R)**2) + g_minus(R)/((R)**3))
                H_m = E_e
                E_m = -H_e

                return hc.Field(E_e, H_e)
        
    
            else:
                null = hc.Vector3d(0.0, 0.0, 0.0)
                return hc.Field(null, null)
        self.E_dipole = E_dipole

        def H_dipole(x, y, z):
            """[Returns a pyHiChi electric and magnetic field object of magnetic dipole geometry at coordinates (x,y,z) in space.]

            :param x,y,z: coordinates in the simulation domain
            :type x,y,z: float
            :return: electric and magnetic pyHiChi field objects at coordinates (x,y,z) of magnetic dipole geometry
            :rtype: pyHiChi field object
            """
            R = np.sqrt(x**2+y**2+z**2)
            if (R>1e-5):
                hhat = hc.Vector3d(1, 0, 0) #X-dir
                n_hat = hc.Vector3d(x, y, z)/R
                h1 = -hc.cross(n_hat, self.d0*hhat)
                e1 = hc.cross(h1, n_hat)
                e2 = 3*hc.dot(n_hat, self.d0*hhat)*n_hat - self.d0*hhat

                H_e = h1*(gpp_plus(R)/(self.c*self.c*(R)) + gp_minus(R)/(self.c*(R)**2))
                E_e = e1*(gpp_minus(R)/((R)*self.c**2))+e2*(gp_plus(R)/(self.c*(R)**2) + g_minus(R)/((R)**3))
                H_m = E_e
                E_m = -H_e

                E_bidip = E_e+E_m
                H_bidip = H_e+H_m
                return hc.Field(E_m, H_m)
        
    
            else:
                null = hc.Vector3d(0.0, 0.0, 0.0)
                return hc.Field(null, null)
        self.H_dipole = H_dipole
        
        def bi_dipole(x, y, z):
            """[Returns a pyHiChi electric and magnetic field object of bi-dipole geometry at coordinates (x,y,z) in space.]

            :param x,y,z: coordinates in the simulation domain
            :type x,y,z: float
            :return: electric and magnetic pyHiChi field objects at coordinates (x,y,z) of bi-dipole geometry
            :rtype: pyHiChi field object
            """
            R = np.sqrt(x**2+y**2+z**2)
            if (R>1e-5):
                ehat = hc.Vector3d(0, 1, 0) #Y-dir
                hhat = hc.Vector3d(1, 0, 0) #X-dir
                n_hat = hc.Vector3d(x, y, z)/R


                #E-dipole
                h1 = -hc.cross(n_hat, self.d0*ehat)
                e1 = hc.cross(h1, n_hat)
                e2 = 3*hc.dot(n_hat, self.d0*ehat)*n_hat - self.d0*ehat
                H_ey = h1*(gpp_plus(R)/(self.c*self.c*(R)) + gp_minus(R)/(self.c*(R)**2))
                E_ey = e1*(gpp_minus(R)/((R)*self.c**2))+e2*(gp_plus(R)/(self.c*(R)**2) + g_minus(R)/((R)**3))

                #H-dipole
                h1 = -hc.cross(n_hat, self.d0*hhat)
                e1 = hc.cross(h1, n_hat)
                e2 = 3*hc.dot(n_hat, self.d0*hhat)*n_hat - self.d0*hhat
                H_ex = h1*(gpp_plus(R)/(self.c*self.c*(R)) + gp_minus(R)/(self.c*(R)**2))
                E_ex = e1*(gpp_minus(R)/((R)*self.c**2))+e2*(gp_plus(R)/(self.c*(R)**2) + g_minus(R)/((R)**3))
                H_mx = E_ex
                E_mx = -H_ex

                return hc.Field(E_mx+E_ey, H_mx+H_ey)
    
            else:
                null = hc.Vector3d(0.0, 0.0, 0.0)
                return hc.Field(null, null)


        self.bi_dipole = bi_dipole

        def init(field_geometry='bi_dipole'):
            """[Initiates a pyHiChi field object that can be propagated within the simulation domain]

            :param field_geometry: string that indicate which field geometry to use, available options: 'e_dipole','h_dipole','bi_dipole'
            :type field_geometry: string
            :return:  a pyHiChi field object that can be propagated within the simulation domain with specified geometry
            :rtype: pyHiChi field object
            """
            if field_geometry == 'bi_dipole':
                field = hc.PSATDPoissonField(self.grid_size, self.min_coords, self.grid_step, self.t_aux)
                field.set(bi_dipole)
                return field
            
            elif field_geometry == 'e_dipole':
                field = hc.PSATDPoissonField(self.grid_size, self.min_coords, self.grid_step, self.t_aux)
                field.set(E_dipole)
                return field
            
            elif field_geometry == 'h_dipole':
                field = hc.PSATDPoissonField(self.grid_size, self.min_coords, self.grid_step, self.t_aux)
                field.set(H_dipole)
                return field
            else:
                print('ERROR : Non-recognized field geometry!')
                return 1
        self.init = init

        def update_field(field, timestep=self.t_aux):
            """[Propagates a pyHiChi field object in time using a specified timestep]

            :param field: pyHiChi field object to be propagated in time
            :type field: pyHiChi field object
            :param timestep: the timestep used to update the field
            :type timestep: float
            :return: 0, pyHiChi field objects are stored in working memory
            :rtype: int
            """
            field.change_time_step(timestep)
            field.update_fields()
            return 0
        self.update_field = update_field

        def get_chi_gamma(ensemble, field, ptype=hc.ELECTRON):
            """[Retrieves the dimensionless acceleration and Lorentz factor of the electron beam for a given pyHiChi field]

            :param ensemble: pyHiChi ensemble with the collection of particles within the beam, default is electrons
            :type ensemble: pyHiChi ensemble object
            :param field: pyHiChi field that is used to compute the value of chi
            :type field: pyHiChi field object
            :param ptype: type of particles within the beam, default is hc.ELECTRON
            :type ptype: pyHiChi ParticleTypes
            :return: tuple of arrays containing the value of chi and gamma for all particles within the beam
            :rtype: tuple of arrays
            """
            E_schwing = (hc.ELECTRON_MASS**2 * hc.c**3)/(-hc.ELECTRON_CHARGE*hc.PLANCK)
            gamma = np.array([el.get_gamma() for el in ensemble[ptype]])
            e = np.array([ field.get_E(el.get_position()) for el in ensemble[ptype]])
            b = np.array([ field.get_B(el.get_position()) for el in ensemble[ptype]])
            v = np.array([ el.get_velocity() for el in ensemble[ptype]])
            chi = lambda gamma,E,B,V : gamma*np.sqrt((  E+hc.cross(V,B)/hc.c  ).norm()**2 - (hc.dot(E,V)/hc.c)**2 )/E_schwing
            XI = np.array([chi(gamma[i],e[i],b[i],v[i]) for i in range(ensemble[ptype].size())])
            return XI, gamma         
        self.get_chi_gamma = get_chi_gamma

            
        def run_sfqed(field, beam, track_diagnostic=None, revert_field=False, k1=0.0, k2=0.0):
            """[Runs the interaction between a given field and beam configuration then returns properties of the interaction depending on the setting specified by track_diagnostic parameter]
            :param field: initial pyHiChi field pbject participating in the interaction
            :type field: pyHiChi field object
            :param beam: particle beam that participates in the interaction
            :type beam: pyHiChi ensemble object
            :param track_diagnostic: string that dictates what property to return, default is to return the particle beam only
            :type track_diagnostic: string
            :param revert_field: propagates the field backwards in time to the original position at the end of the simulation
            :type revert_field: string
            :param k1: will be removed since it plays no role in the code
            :type k1: float
            :param k2: will be removed since it plays no role in the code
            :type k2: float
            :return: beam, photon_count if 'photons' : n_single_events if 'events' : beam if else
            :rtype: pyHiChi ensemble object, float : float : pyHiChi ensemble object
            """
            if track_diagnostic == 'photons':
                photon_count = []
                for _ in range(self.n_iter):
                    self.qed.process_particles(beam, field, self.time_fine_step, k1, k2)
                    update_field(field, timestep=self.time_fine_step)
                    photon_count.append(beam['Photon'].size())
                
                if revert_field:
                    update_field(field, self.dt_revert)

                return beam, photon_count

            elif track_diagnostic == 'events':
                threshold = 1.0 #Multiple of the electron rest mass; is DeltaGamma > mult [in units of rest mass]?
                n_particles = beam[hc.ELECTRON].size()
                chi_max = self.chi_max
                chi_max_array = chi_max*np.ones(n_particles)
                emissions = np.zeros(n_particles)
                for _ in range(self.n_iter):
                    chi_1, gam_1 = get_chi_gamma(beam, field)
                    self.qed.process_particles(beam, field, self.time_fine_step, k1, k2)
                    update_field(field, self.time_fine_step)
                    chi_2, gam_2 = get_chi_gamma(beam, field)
                    bool_array_chi = chi_1 > chi_max_array
                    bool_array_gamma = gam_2 < gam_1-threshold
                    bool_array_AND = np.array(np.logical_and(bool_array_chi, bool_array_gamma))
                    emissions += bool_array_AND
                emissions = np.array(emissions)
                n_single_events = np.count_nonzero(emissions == 1)
                if revert_field:
                    update_field(field, self.dt_revert)

                return n_single_events
                
            else:
                if revert_field:
                    update_field(field, self.dt_revert)
                return beam
        self.run_sfqed = run_sfqed

        def generate_points(length, radius, distribution='normal'):
            """[Generates beam coordinate points uniformally in the beam propagation direction and circularly or normally distributed in the transverse direction depending on the distribution parameter]
            :param length: length of the particle beam
            :type length: float
            :param radius: effective radius of the transverse distributed area
            :type radius: float
            :param distribution: setting to change how the transverse coordinates are distributed
            :type distribution: string
            :return: beam coordinate points according to specified transverse distribution
            :rtype: tuple of floats
            """
            if distribution == 'normal':
                spread = radius/(np.sqrt(2*np.log(2))) 
                y, z = np.random.normal(0.0,spread), np.random.normal(0.0,spread)
                return (length/2)*np.random.uniform(-1,1), y, z
            elif distribution == 'uniform':
                length_circ = np.sqrt(np.random.uniform(0,1))
                angle = np.pi*np.random.uniform(0,2)
                return (length/2)*np.random.uniform(-1,1), length_circ*np.cos(angle)*radius, length_circ*np.sin(angle)*radius
            else:
                print('Unknown distribution, check PyPole.py')
                return 0
        self.generate_points = generate_points

        def uniform_electron_beam(input_erg=self.electron_energy, no_eons=self.number_e, distribution='uniform'):
            """[Creates a particle beam using a specified distribution]
            :param input_erg: input energy for each particle in the beam
            :type input_erg: float
            :param no_eons: number of simulated particles within the bea,
            :type no_eons: int
            :param distribution: how the particles are distributed in a direction transverse to the propagation direction in space
            :type distribution: string
            :return: particle beam with coordinate points according to specified transverse distribution
            :rtype: pyHiChi ensemble object
            """
            beam = hc.Ensemble()
            if distribution == 'normal':
                while(beam[hc.ELECTRON].size() < no_eons):
                    x, y, z = generate_points(beam_length, spot_radius, distribution=distribution)
                    randcoord = hc.Vector3d(self.distance_start+x, y, z)
                    if hp.block(randcoord.y, self.min_coords.y, self.max_coords.y) == 0 or hp.block(randcoord.z, self.min_coords.z, self.max_coords.z) == 0:
                        continue
                    dE = input_erg*(1+0.001*np.random.random())
                    mo_val = np.sqrt(((dE)/self.c)**2 - (self.me*self.c)**2)
                    mo = hc.Vector3d(-mo_val, 0.0, 0.0)        
                    particle = hc.Particle(randcoord, mo, self.weight, hc.ELECTRON)
                    beam.add(particle)

            elif distribution == 'uniform':
                for _ in range(no_eons):
                    y, x, z = generate_points(self.beam_length, self.spot_radius, distribution=distribution)
                    randcoord = hc.Vector3d(x, self.distance_start+y, z)
                    dE = input_erg*(1+0.001*np.random.random())
                    mo_val = np.sqrt(((dE)/self.c)**2 - (self.me*self.c)**2)
                    mo = hc.Vector3d(0.0, -mo_val, 0.0)        
                    particle = hc.Particle(randcoord, mo, self.weight, hc.ELECTRON)
                    beam.add(particle)
            
            return beam
        self.uniform_electron_beam = uniform_electron_beam

        def get_E_norm(field, shape=(1024,1024), orientation='xy'):
            """[Retrieves the norm of the electric field on a grid that is located in the simulation domain origin oriented in space specified by the user]
            :param field: the field object to retrieve the norm of the electric field from
            :type field: pyHiChi field object
            :param shape: resolution or shape of the grid that the norm will be sampled upon
            :type shape: tuple of floats of length 2
            :param orientation: setting that specifies how the grid sampling is done in space, default is 'xy' which gives the norm in the xy-plane. Options available; 'xy','yz','xz'
            :type orientation: string
            :return: grid with sampled electric field norm for the pyHiChi field object input
            :rtype: Ndarray
            """
            res = np.zeros(shape=(shape[0], shape[1]))
            if orientation == 'xy':
                step = ((self.max_coords.x - self.min_coords.x)/shape[0], (self.max_coords.y - self.min_coords.y)/shape[1])
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        x = self.min_coords.x + i*step[0]
                        y = self.min_coords.y + j*step[1]
                        coord = hc.Vector3d(x,y,0.0)

                        res[shape[1]-j-1, i] = field.get_E(coord).norm()

            if orientation == 'yz':
                step = ((self.max_coords.y - self.min_coords.y)/shape[0], (self.max_coords.z - self.min_coords.z)/shape[1])
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        y = self.min_coords.y + i*step[0]
                        z = self.min_coords.z + j*step[1]
                        coord = hc.Vector3d(0.0,y,z)

                        res[shape[1]-j-1, i] = field.get_E(coord).norm()

            elif orientation == 'xz':
                step = ((self.max_coords.x - self.min_coords.x)/shape[0], (self.max_coords.z - self.min_coords.z)/shape[1])
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        x = self.min_coords.x + i*step[0]
                        z = self.min_coords.y + j*step[1]
                        coord = hc.Vector3d(x,0.0,z)

                        res[shape[1]-j-1, i] = field.get_E(coord).norm()
            
            return res
        self.get_E_norm = get_E_norm

        def field_image(field, parray=None, shape=(1024, 1024), orientation='xy', cmap='YlOrRd', fwidth=3.38583, fsize=12, show_beam=False, save_fig=False, alp=0.2):
            """[Visualizes the electric field norm for a field with a particle beam (beam optional) on a specified grid oriented in space and centered on the origin of the simulation domain]
            :param field: pyhiChi field object to be visualized
            :type field: pyHiChi field object
            :param parray: particle beam to be visualized, default is None to call the function quickly
            :type parray: None or pyHiChi ensemble object
            :param shape: the shape of the sampling grid for the norm of the pyHiChi field object
            :type shape: tuple of length 2
            :param orientation: setting that specifies to visualize the field and beam geometry in space, default is 'xy' which gives the norm in the xy-plane. Options available; 'xy','yz','xz'
            :type orientation: string
            :param cmap: colormap used for visualization of the image, default is 'YlOrRd'
            :type cmap: string
            :param fwidth: width of the plot figure in inches
            :type fwidth: float
            :params fsize: fontsize of labels in the figure
            :type fsize: int
            :param show_beam: setting if the particle beam should be shown or not
            :type show_beam: bool
            :param save_fig: option to save the figure
            :type save_fig: bool
            :param alp: transparency of the visualized particles in the beam, 0 renders them invisible and 1 makes them fully visible
            :type alp: float
            :return: 0, only a function to visualize the simulation geometry
            :rtype: int
            """
            golden_ratio = (1+np.sqrt(5))/2
            fig = plt.figure(figsize=(fwidth, fwidth*golden_ratio))
            ax = fig.add_subplot(111)

            ptype=hc.ELECTRON

            if orientation == 'xy':
                values = get_E_norm(field, shape=shape, orientation='xy')
                values = (values - values.min())/(values.max() - values.min())
                im = ax.imshow(values, extent=(self.min_coords.x/self.wavelength, self.max_coords.x/self.wavelength, self.min_coords.y/self.wavelength, self.max_coords.y/self.wavelength), cmap=cmap)
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                XLABEL = 'x'+' '+'$[\\mu m]$'
                YLABEL = 'y'+' '+'$[\\mu m]$'
                TITLE = 'Cross sectional plot of electric field strength'
                if show_beam:
                    x, y = [], []
                    for el in parray[ptype]:
                        x.append(el.get_position().x/self.wavelength)
                        y.append(el.get_position().y/self.wavelength)
                    ax.scatter(x, y, color='blue', marker='o', s=15, edgecolors='black', label='Electrons', alpha=alp)
                    ax.legend()
                

            elif orientation == 'yz':
                values = get_E_norm(field, shape=shape, orientation='yz')
                values = (values - values.min())/(values.max() - values.min())
                im = ax.imshow(values, extent=(self.min_coords.y/self.wavelength, self.max_coords.y/self.wavelength, self.min_coords.z/self.wavelength, self.max_coords.z/self.wavelength), cmap=cmap)
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                XLABEL = 'y'+' '+'$[\\mu m]$'
                YLABEL = 'z'+' '+'$[\\mu m]$'
                TITLE = 'Electric field strength in the yz-plane'
                if show_beam:
                    y, z = [], []
                    for el in parray[ptype]:
                        y.append(el.get_position().y/self.wavelength)
                        z.append(el.get_position().z/self.wavelength)
                    ax.scatter(y, z, color='blue', marker='o', s=15, edgecolors='black', label='Electrons', alpha=alp)
                    ax.legend()


            elif orientation == 'xz':
                values = get_E_norm(field, orientation='xz')
                values = (values - values.min())/(values.max() - values.min())
                im = ax.imshow(values, extent=(self.min_coords.x/self.wavelength, self.max_coords.x/self.wavelength, self.min_coords.z/self.wavelength, self.max_coords.z/self.wavelength), cmap=cmap)
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                XLABEL = 'x'+' '+'$[\\mu m]$'
                YLABEL = 'z'+' '+'$[\\mu m]$'
                TITLE = 'Electric field strength in the xz-plane'
                if show_beam:
                    x, z = [], []
                    for el in parray[ptype]:
                        x.append(el.get_position().x/self.wavelength)
                        z.append(el.get_position().z/self.wavelength)
                    ax.scatter(x, z, color='blue', marker='o', s=15, edgecolors='black', label='Electrons', alpha=alp)
                    ax.legend()

            ax.set_xlabel(XLABEL, fontsize=fsize)
            ax.set_ylabel(YLABEL, fontsize=fsize)
            ax.set_title(TITLE, fontsize=fsize)
            ax.tick_params(axis='both', labelsize=fsize)
            cbar.set_label('Electric field strength [arb. units]', fontsize=fsize)
            fig.tight_layout()

            if save_fig:
                name = input()
                plt.savefig(name)
            plt.show()

            return 0
        self.field_image = field_image


        def PE_imshow(data, parray, earray, chi_maxes, cmap='Wistia', fsize=14, fwidth=3.38583, log_scale=False, save_fig=False):
            """[Generates a heatmap with the parametric scan over input power and beam energy as a function of the event rate]
            :param data: array of 2D arrays to be displayed, one for each chi treshold
            :type data: array of 2D arrays
            :param parray: array with linearly spaced input power values that is included in the parametric scan
            :type parray: array of floats
            :param earray: array with linearly spaced beam energy values that is included in the parametric scan
            :type earray: array of floats
            :param chi_maxes: array of chi thresholds to be considered for each heatmap
            :type chi_maxes: array of floats
            :param cmap: colormap used for visualization of the image, default is 'Wistia'
            :type cmap: string
            :params fsize: fontsize of labels in the figure
            :type fsize: int
            :param fwidth: width of the plot figure in inches
            :type fwidth: float
            :param log_scale: setting to plot the heatmaps in logscale
            :type log_scale: bool
            :param save_fig: option to save the figure
            :type save_fig: bool
            :return: 0, only a function to visualize the heatmaps of parametric scans
            :rtype: int
            """
            #misc
            golden_ratio = (1+np.sqrt(5))/2
            peta_conv = 1e+22
            gev_conv = 0.00160218

            #fig_data
            n_maps = len(data)
            figure_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
            XLABEL = 'Electron energy [GeV]'
            YLABEL = 'Average input power [PW]'
            CBAR_LABEL = '$f_{\chi}$ [%]'


            #fig plotting
            fig, axs = plt.subplots(nrows=1, ncols=n_maps, figsize=(fwidth, fwidth*golden_ratio), sharey=True)
            if log_scale:
                [(axs[j].set_yscale('log'), axs[j].set_xscale('log')) for j in range(n_maps)]

            for i in range(n_maps):
                if i == 0:
                    axs[i].set_ylabel(YLABEL, fontsize=fsize)

                im = axs[i].imshow(np.flip(data[i], axis=0), extent=(earray[0]/gev_conv, earray[-1]/gev_conv, parray[0]/peta_conv, parray[-1]/peta_conv), cmap=cmap)
                EXTRA_LABEL = 'Events registered above $\\chi >$'+str(chi_maxes[i])
                axs[i].text(0.05, 0.05, EXTRA_LABEL, transform=axs[i].transAxes, fontsize=fsize*0.65, backgroundcolor='white')
                axs[i].set_xlabel(XLABEL, fontsize=fsize)
                
            
            axin = inset_axes(axs[-1],
                            width="10%",  # width = 50% of parent_bbox width
                            height="100%",  # height : 5%
                            loc='center right',
                            borderpad=-4)
            cbar = fig.colorbar(im, cax=axin)
            cbar.set_label(CBAR_LABEL, fontsize=fsize)
            fig.tight_layout()

            if save_fig:
                name = input()
                plt.savefig(name)

            plt.show()

            return 0
        self.PE_imshow = PE_imshow