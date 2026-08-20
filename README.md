# Solar Flare Forecaster
Most solar flare forecasters use traditional machine learning models such as XGBoost. These ML models are fed images and histories of the sun's photosphere (lowest area you can take an image of on the sun), to forecast if and what type of solar flare would occur. This project attempts to take an attempt at using fundamental physics laws such as those involving Maxwell's equations and Magnetohydrodynamics. These equations are given to a PINN in the hopes that we can understand the curl or twist of the fields. By mapping this twisting we can get a better view of active regions where solar flares may occur. This project is currently underway and I'm actively trying out different branches/prototypes so stay tuned for more results!

To give you an idea of what we're trying to generate here are some ideas below:

<img width="251" height="277" alt="Screenshot 2026-08-20 at 12 56 19 PM" src="https://github.com/user-attachments/assets/f24dcc8c-72c1-4988-b001-a03c18501a51" />

This is the twist, or the curl of the magnetic field on the sun from an image

<img width="248" height="268" alt="Screenshot 2026-08-20 at 12 56 47 PM" src="https://github.com/user-attachments/assets/de1c2a86-74da-40b6-9751-cd1b0c1682de" />

This is the magnetic field plotted by strength (more red) on the sun based on the curl image

<img width="754" height="317" alt="Screenshot 2026-08-20 at 1 06 27 PM" src="https://github.com/user-attachments/assets/081f1876-4468-4fa4-879c-8f71ae8434aa" />

To give you a super simple idea of what I think our pipeline can look like at inference this is the current model I'm trying out. Note that I'm leaving out a lot of the training components and small details
