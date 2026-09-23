"""Exact moment-matched smooth laws with opposite fixed harm judgments."""
from fractions import Fraction
from math import comb
from pathlib import Path
import json
import sys


def uniform_band_moment(center, half_width, degree):
    return sum(Fraction(comb(degree,noise_degree))*center**(degree-noise_degree)*half_width**noise_degree/Fraction(noise_degree+1)
        for noise_degree in range(0,degree+1,2))


def main(directory):
    configuration=json.loads((directory/'config.json').read_text());rows=[];checks=0
    for moment_count in configuration['moment_counts']:
        difference_order=moment_count+1
        if difference_order%2==0:
            difference_order+=1
        normalizer=2**(difference_order-1)
        half_width=Fraction(1,8*difference_order)
        shifts=[-Fraction(1,16*difference_order),Fraction(1,16*difference_order)]
        forecast_vectors=[];risks=[]
        for shift in shifts:
            laws=[]
            for parity in (0,1):
                law=[(Fraction(1,4)+Fraction(index,2*difference_order)+shift,
                    Fraction(comb(difference_order,index),normalizer))
                    for index in range(difference_order+1) if index%2==parity]
                assert sum(weight for center,weight in law)==1
                assert all(0<center-half_width<center+half_width<1 for center,weight in law)
                assert all(abs(center-Fraction(1,2))>half_width for center,weight in law)
                laws.append(law)
            vectors=[[sum(weight*uniform_band_moment(center,half_width,degree) for center,weight in law)
                for degree in range(1,moment_count+1)] for law in laws]
            assert vectors[0]==vectors[1]
            checks+=moment_count
            hazard_probabilities=[sum(weight for center,weight in law if center>Fraction(1,2)) for law in laws]
            assert sum(hazard_probabilities)==1
            assert (hazard_probabilities[0]>Fraction(1,2))!=(hazard_probabilities[1]>Fraction(1,2))
            expected_gap=Fraction(comb(difference_order-1,(difference_order-1)//2),normalizer)
            assert abs(hazard_probabilities[0]-hazard_probabilities[1])==expected_gap
            forecast_vectors.append(vectors[0]);risks.append(hazard_probabilities)
        assert risks[0]==risks[1]
        baseline=[(first+second)/2 for first,second in zip(*forecast_vectors)]
        useful_gain=sum((value-center)**2 for vector in forecast_vectors for value,center in zip(vector,baseline))/Fraction(2*moment_count)
        assert useful_gain>0
        # Conditional mean forecasts of every Y**j survive erasing the independent parity bit.
        # Every prior-mean decision reader sees identical forecast vectors under both parities.
        reader_checks=0
        for first in range(configuration['reader_denominator']+1):
            for second in range(configuration['reader_denominator']+1):
                predictions=[Fraction(first,configuration['reader_denominator']),Fraction(second,configuration['reader_denominator'])]
                accuracy=sum(Fraction(1,4)*(prediction if risk>Fraction(1,2) else 1-prediction)
                    for prediction,probabilities in zip(predictions,risks) for risk in probabilities)
                assert accuracy==Fraction(1,2)
                reader_checks+=1
        probability_error=expected_gap**2/4
        rows.append({'moment_count':moment_count,'difference_order':difference_order,
            'risk_probabilities':[str(value) for value in risks[0]],'risk_margin':str(expected_gap/2),
            'uniform_band_half_width':str(half_width),'useful_signal_gain':str(useful_gain),
            'useful_signal_retained':'1','judgment_accuracy_after_erasure':'1/2',
            'risk_probability_mse_after_erasure':str(probability_error),
            'matched_moment_vectors':[[str(value) for value in vector] for vector in forecast_vectors],
            'reader_grid_checks':reader_checks})
    assert next(row for row in rows if row['moment_count']==2)['risk_margin']=='1/4'
    # Adding the event indicator itself distinguishes the laws, as the affine-span criterion requires.
    report={'status':'complete','moment_equalities':checks,'families':len(rows),
        'randomized_reader_checks':sum(row['reader_grid_checks'] for row in rows),
        'continuous_density_laws':True,'fixed_harm_threshold':'1/2','fixed_risk_tolerance':'1/2',
        'event_feature_control_distinguishes_all':all(row['risk_margin']!='0' for row in rows),
        'general_SCC_impossibility':False,'neural_training':False,'results':rows}
    (directory/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='results'}))
    for row in rows:
        print(json.dumps({key:value for key,value in row.items() if key!='matched_moment_vectors'}))

if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
