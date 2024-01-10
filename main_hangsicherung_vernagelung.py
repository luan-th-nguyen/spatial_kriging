import math
import numpy as np
from PIL import Image

def main_hangsicherung_vernagelung(st):
    st.markdown('Nachweis der lokalen Standsicherheit nach der Methode RUVOLUM')
    st.markdown('Cala M, Flum D, Roduner A, Ruegger R, Wartmann ST. Slope Stabilization System and RUVOLUM® dimensioning method. AGH University of Science and Technology, Faculty of Mining and Geoengineering. 2012.')

    st.header('Project information')
    col1, col2, col3 = st.columns(3)
    col1.text_input('BV', value='Erneuerung Hangsicherung')
    col2.text_input('Strecke', value='DB Strecke 5310')
    col3.text_input('Bereich', value='Bereich 3: natürliche Felsböschung')

    st.header('Systemeigenschaften')
    st.markdown('**zum System**')
    # Geflechttyp
    # Krallplatte
    col1, col2 = st.columns(2)
    ah = col1.number_input('Horizontaler Abstand ah [m]', value=2.5)
    av = col2.number_input('Abstand in Hangsrichtung av [m]', value=2.5)
    col1, col2, col3 = st.columns(3)
    col1.text_input('Nageltyp', value='TITAN 30/11')
    nagel_psi = col2.number_input('Nagelneigung psi [°]', value=45.0)
    nagel_l = col3.number_input('Nagellänge (ohne Überstand) l [m]', value=3.6)
    col1.text_input('Geflechttyp', value='GEOBRUGG Tecco G65/3')

    st.markdown('**zum Stahldrahtgeflecht**')
    col1, col2, col3 = st.columns(3)
    zRkv = col1.number_input('Zugfestigkeit in Längsrichtung z_Rkv[kN/m]', value=150.0)
    zRkh = col2.number_input('Zugfestigkeit in Querrichtung z_Rkh [kN/m]', value=50.0)
    ZRk = col3.number_input('Tragwiderstand auf punktuelle, böschungsparallele Zugbeanspruchung Z_R [kN]k', value=30.0)
    DRk = col1.number_input('Tragwiderstand gegen Durchstanzen in Nagelrichtung D_Rk [kN]', value=180.0)
    PRk = col2.number_input('Tragwiderstand gegen Abscheren am Krallplattenrand P_Rk [kN]', value=90.0)
    Zd = col1.number_input('Nach oben gerichtete, böschungsparallele Kraft im Netz (Standardwert nach Geobrugg) Zd [kN]', value=15.0)
    V = col2.number_input('Systemvorpannkraft (Standardwert nach Geobrugg) V [kN]', value=30.0)
    zeta = col2.number_input('Oberer Radius des Druckkegels zeta [m]', value=0.17)
    delta = col2.number_input('Neigung des Druckkegels delta [°]', value=45.0)

    st.markdown('**zum Nagel**')
    col1, col2, col3, col4 = st.columns(4)
    nagel_Da_datenblatt = col1.number_input('Außendurchmesser (inkl. Schraubrippen) DA [mm]', value=29.0)
    nagel_Di_datenblatt = col2.number_input('Innendurchmesser DI [mm]', value=13.0)
    nagel_As_datenblatt = col3.number_input('Querschnittsfläche As [mm^2]', value=415.0)
    nagel_Rk_datenblatt = col4.number_input('Charakteristische Tragfähigkeit Rk [kN]', value=255.0)
    nagel_delta_t_outside = col1.number_input('Abrostung, außenseitig [mm]', value=4.0)
    nagel_concrete_cover = col2.number_input('Betondeckung [mm]', value=30.0)
    Da_uncorroded = math.sqrt(nagel_As_datenblatt*4.0/math.pi + nagel_Di_datenblatt**2)
    fyk = nagel_Rk_datenblatt*1000.0/nagel_As_datenblatt   # N/mm^2
    Da_corroded = Da_uncorroded - nagel_delta_t_outside
    As_statik = (Da_corroded**2 - nagel_Di_datenblatt**2)*math.pi/4.0
    Rk_statik = As_statik * fyk * 0.001 # kN
    Rqk_statik = As_statik * fyk/math.sqrt(3) * 0.001 # kN
    col1, col2 = st.columns(2)
    col1.write('Statisch relevanter Außendurchmesser, unabgerostet Da = {0:.2f} [mm]'.format(Da_uncorroded))
    col2.write('Charakteristische Festigkeit fy,k = Rk / As = {0:.2f} [N/mm^2]'.format(fyk))
    col1.write("Statisch relevanter Außendurchmesser, abgerostet Da' = {0:.2f} [mm]".format(Da_corroded))
    col2.write("Ansetzbare Querschnittsfläche As' = {0:.2f} [mm^2]".format(As_statik))
    col1.write("Char. Tragfähigkeit unter Abrostung, Zugbeanspruchung R'k  = {0:.2f} [kN]".format(Rk_statik))
    col2.write("Char. Tragfähigkeit unter Abrostung, Schubbeanspruchung R'q,k   = {0:.2f} [kN]".format(Rqk_statik))

    st.markdown('**Charakteristische Bodenkennwerte**')
    col1, col2, col3, col4 = st.columns(4)
    alpha_k = col1.number_input('Böschungsneigung alpha_k [°]', value=45.0)
    phi_k = col2.number_input('Reibungswinkel phi_k [°]', value=30.0)
    c_k = col3.number_input('Kohäsion c_k [kN/m^2]', value=0.0)
    wichte_k = col4.number_input('Wichte gamma_k [kN/m^3]', value=22.0)
    wichte_prime_k = col1.number_input("Wichte unter Auftrieb gamma_k' [kN/m^3]", value=12.0)
    qs_k = col2.number_input('Mantelreibung qs_k [kN/m^2]', value=200.0)
    t = col3.number_input('Dicke der aufgelockerten/ verwitterten Schicht t [m]', value=0.6)

    st.markdown('**Sicherheitsbeitwerte Nachweisverfahren GEO-3 (BS-P)**')
    col1, col2, col3, col4 = st.columns(4)
    gamma_phi = col1.number_input('Reibungswinkel gamma_phi [-]', value=1.25)
    gamma_c = col2.number_input('Kohäsion gamma_c [-]', value=1.25)
    gamma_G = col3.number_input('Eigengewicht gamma_G [-]', value=1.0)
    gamma_M = col4.number_input('Teisicherheitsbeiwert Stahlzugglied gamma_M [-]', value=1.15)
    gamma_st = col1.number_input('Pfahlmantelwiderstand (Zug) aus Pfahlprobebelastung gamma_st [-]', value=1.15)
    eta_M = col2.number_input('Modellfaktor für verpresste Mikropfähle (DIN 1054) ηM [-]', value=1.25)
    gamma_mod = col3.number_input('Korrekturwert der Modellunsicherheit (gem. RUVOLUM) gamma_mod [-]', value=1.1)

    st.markdown('**Bemessungswerte der Bodenkennwerte**')
    phi_d = math.atan(math.tan(phi_k*math.pi/180)/gamma_phi)*180/math.pi
    c_d = c_k/gamma_c
    wichte_d = wichte_k/gamma_G
    wichte_prime_d = wichte_prime_k/gamma_G

    col1, col2, col3, col4 = st.columns(4)
    col1.write('Reibungswinkel phi_d = {0:.2f} [°]'.format(phi_d))
    col2.write('Kohäsion c_d = {0:.2f} [°]'.format(c_d))
    col3.write('Wichte gamma_d = {0:.2f} [°]'.format(wichte_d))
    col4.write("Wichte unter Auftrieb gamma_d' = {0:.2f} [°]".format(wichte_prime_d))


    st.header('Nachweise')
    st.markdown('**Bruchmechanismus 1: Oberflächennahe, böschungsparallele Instabilitäten**')
    image_falure_mode1 = Image.open('./media/nail_local_failure_mode1.PNG')
    st.image(image_falure_mode1)
    col1, col2, col3, col4 = st.columns(4)
    Vdl = col1.number_input('Bemessungswert der aufgebrachten Systemvorspannkraft (günstig) Vdl [kN]', value=24.0)
    Vdll = col2.number_input('Bemessungswert der aufgebrachten Systemvorspannkraft (ungünstig) Vdll [kN]', value=30.0)
    Gd = ah*av*t*wichte_prime_d
    Fs = ah*av*t*10.0*math.sin(alpha_k*math.pi/180)
    x = (Gd*math.cos(alpha_k*math.pi/180) + Vdl*math.sin((alpha_k + nagel_psi)*math.pi/180))*math.tan(phi_d*math.pi/180 - ah*av*c_d)/gamma_mod
    Sd = Fs + Gd*math.sin(alpha_k*math.pi/180) - Vdl*math.cos((alpha_k + nagel_psi)*math.pi/180) - x
    col3.write('Bemessungswert der Gewichtskraft des Bruchkörpers Gd = {0:.2f} [kN]'.format(Gd))
    col4.write('Strömungskraft Fs = {0:.2f} [kN]'.format(Fs))
    st.write("Bemessungswert der Schubbeanspruchung des Nagels Sd = {0:.2f} [kN]".format(Sd))

    st.markdown('**Bruchmechanismus 2: lokale Instabilitäten zwischen den einzelnen Nägeln**')
    image_falure_mode2 = Image.open('./media/nail_local_failure_mode2.PNG')
    st.image(image_falure_mode2)

    st.markdown('**Fall 2A: Einkörper-Gleitmechanismus**')
    image_falure_mode2A = Image.open('./media/nail_local_failure_mode2A.PNG')
    st.image(image_falure_mode2A)
    ah_red = get_ah_reduced(ah, zeta, t, delta)
    st.write('Reduzierte Breite des Bruchkörpers ah_red = {0:.2f} [m]'.format(ah_red))
    beta = 38.16 # deg.
    r = get_geometric_parameters_1body(beta, alpha_k, ah_red, av, wichte_prime_d, t)
    st.write('maßgebende Schichtdicke ti = {0:.2f} [m]'.format(r[0]))
    st.write('Bemessungswert der Gewichtskraft des Bruchkörpers Gdi = {0:.2f} [m]'.format(r[1]))
    #Pd1 = calc_Pd1(Gd, Fs, Zd, alpha_k, beta, nagel_psi, phi_d, c_d, ah*av, gamma_mod)
    #st.write('Abscherbeanspruchung des Gefelchtes am unteren Krallplattenrand Pd1 = {0:.2f} [kN]'.format(Pd1))




    st.markdown('**Fall 2B: Zweikörper-Gleitmechanismus**')
    image_falure_mode2B = Image.open('./media/nail_local_failure_mode2B.PNG')
    st.image(image_falure_mode2B)


def get_ah_reduced(ah, zeta, t, delta=45.0):
    """ Gets reduced horizontal spacing between soil nails
    """
    ah_red =  ah - 2*zeta - t/math.tan(delta*math.pi/180)
    return ah_red


def get_geometric_parameters_1body(beta, alpha, ah_red, av, wichte, t):
    """ Gets geometric parameters
    """
    beta_rad = beta*math.pi/180
    alpha_rad = alpha*math.pi/180
    ti = math.tan(alpha_rad - beta_rad) * 2*av
    Gdi = 2*av*ti/2*ah_red*wichte
    #Fs
    #A

    return (ti, Gdi)


def calc_Pd1(Gd, Fs, Zd, alpha, beta, psi, phi_d, c_d, A, gamma_mod):
    alpha_rad = alpha*math.pi/180
    beta_rad = beta*math.pi/180
    psi_rad = psi*math.pi/180
    phi_d_rad = phi_d*math.pi/180
    term1 = Fs*math.cos(alpha_rad - beta_rad) + Gd*math.sin(beta_rad) - Zd*math.cos(alpha_rad - beta_rad)
    term2 = (Gd*math.cos(beta_rad) - Zd*math.sin(alpha_rad - beta_rad) + Fs*math.sin(alpha_rad - beta_rad))*math.tan(phi_d_rad) + c_d*A
    term3 = math.cos(beta_rad + psi_rad) + math.sin(beta_rad + psi_rad)*math.tan(phi_d_rad)/gamma_mod
    r = (term1 - term2)/term3

    return r