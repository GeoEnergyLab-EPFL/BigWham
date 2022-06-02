//
// Created by Alexis Sáez Uribe on 30/05/2022.
//

#include <elasticity/2d/ElasticAxi3DP0_element.h>
#include <src/core/ElasticProperties.h>
#include <elasticity/2d/elliptic_integral.hpp>

namespace bie {

    il::StaticArray2D<double, 2, 2> traction_influence_Axi3DP0(
            SegmentData &source_elt, // source element
            SegmentData &receiver_elt, // receiver element
            bie::ElasticProperties const &elas_) { // elastic properties

        // get constitutive parameters. NOTE:   Poisson's ratio is taken as zero
        double G = elas_.getG();

        double rExt = source_elt.Xmid(0) + source_elt.size()/2.0;
        double rInt = source_elt.Xmid(0) - source_elt.size()/2.0;

        double rObs = receiver_elt.Xmid(0);

        double IF = stress_disk_dislocation(rObs,rExt) - stress_disk_dislocation(rObs,rInt);

        il::StaticArray2D<double, 2, 2> traction_vector; // (shear, normal)

        // shear stress
        //  effect of shear dd
        traction_vector(0, 0) = 2.0 * G * IF;
        //  effect of normal dd
        traction_vector(0, 1) = 0.0;

        // normal stress
        //  effect of shear dd
        traction_vector(1, 0) = 0.0;
        //  effect of normal dd
        traction_vector(1, 1) = 2.0 * G * IF; // 2G=E when nu=0

        return traction_vector;
    }

    double stress_disk_dislocation(
            double rObs, // radial coordinate of the observation point
            double rSrc // radius of the dislocation disk
    ) {
        double IF;
        if(rSrc > 0.0){
            double rho = rObs/rSrc;
            double k = 2.0 * pow(rho,0.5) / (1.0 + rho);
            double k1 = (1.0 - rho) / (1.0 + rho);
            double P101 = (k / ( 2.0 * il::pi * pow(rho,0.5) ) ) *
                          ( ( pow(k,2.0) * (1.0 - pow(rho,2.0)) / (4.0 * rho * pow(k1,2.0)) ) *
                            elliptic_em(pow(k,2.0)) + elliptic_fm(pow(k,2.0)) );
            IF = P101 / ( 4.0 * rSrc );
        }
        else{
            IF = 0.0;
        }

        return IF;
    }
}