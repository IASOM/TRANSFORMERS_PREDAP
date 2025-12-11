import pandas as pd
import numpy as np


codes_agrupations_low_level = {
    "Intestinal infectious diseases": ["A00", "A09"],
    "Tuberculosis": ["A15", "A19"],
    "Certain zoonotic bacterial diseases": ["A20", "A28"],
    "Other bacterial diseases": ["A30", "A49"],
    "Sexually transmitted infections": ["A50", "A64"],
    "Other spirochaetal diseases": ["A65", "A69"],
    "Chlamydial diseases": ["A70", "A74"],
    "Rickettsioses": ["A75", "A79"],
    "Viral infections of the CNS": ["A80", "A89"],
    "Arthropod-borne viral fevers & haemorrhagic fevers": ["A90", "A99"],
    "Other viral diseases": ["B00", "B34"],
    "Viral hepatitis": ["B15", "B19"],
    "HIV disease": ["B20", "B24"],
    "Other human herpesviruses": ["B00", "B09"],
    "Mycoses": ["B35", "B49"],
    "Protozoal diseases": ["B50", "B64"],
    "Helminthiases": ["B65", "B83"],
    "Pediculosis, acariasis & other infestations": ["B85", "B89"],
    "Other infectious diseases": ["B90", "B99"],
    "Malignant neoplasms of lip, oral cavity & pharynx": ["C00", "C14"],
    "Malignant neoplasms of digestive organs": ["C15", "C26"],
    "Malignant neoplasms of respiratory & intrathoracic organs": ["C30", "C39"],
    "Malignant neoplasms of bone & articular cartilage": ["C40", "C41"],
    "Melanoma & other malignant neoplasms of skin": ["C43", "C44"],
    "Malignant neoplasms of mesothelial & soft tissue": ["C45", "C49"],
    "Malignant neoplasm of breast": ["C50", "C50"],
    "Malignant neoplasms of female genital organs": ["C51", "C58"],
    "Malignant neoplasms of male genital organs": ["C60", "C63"],
    "Malignant neoplasms of urinary tract": ["C64", "C68"],
    "Malignant neoplasms of eye, brain & other CNS": ["C69", "C72"],
    "Malignant neoplasms of endocrine glands": ["C73", "C75"],
    "Secondary & unspecified malignant neoplasms": ["C76", "C80"],
    "Malignant neoplasms of lymphoid, haematopoietic & related tissue": ["C81", "C96"],
    "In situ neoplasms": ["D00", "D09"],
    "Benign neoplasms": ["D10", "D36"],
    "Neoplasms of uncertain or unknown behaviour": ["D37", "D48"],
    "Nutritional anaemias": ["D50", "D53"],
    "Haemolytic anaemias": ["D55", "D59"],
    "Aplastic & other anaemias": ["D60", "D64"],
    "Coagulation defects, purpura & other haemorrhagic conditions": ["D65", "D69"],
    "Other diseases of blood & blood-forming organs": ["D70", "D77"],
    "Certain disorders involving the immune mechanism": ["D80", "D89"],
    "Thyroid disorders": ["E00", "E07"],
    "Diabetes mellitus": ["E10", "E14"],
    "Other endocrine disorders": ["E15", "E35"],
    "Malnutrition": ["E40", "E46"],
    "Other nutritional deficiencies": ["E50", "E64"],
    "Obesity & other hyperalimentation": ["E65", "E68"],
    "Metabolic disorders": ["E70", "E90"],
    "Organic mental disorders": ["F00", "F09"],
    "Substance-related disorders": ["F10", "F19"],
    "Schizophrenia & psychotic disorders": ["F20", "F29"],
    "Mood [affective] disorders": ["F30", "F39"],
    "Neurotic, stress-related & somatoform disorders": ["F40", "F48"],
    "Behavioural syndromes": ["F50", "F59"],
    "Disorders of adult personality & behaviour": ["F60", "F69"],
    "Mental retardation": ["F70", "F79"],
    "Disorders of psychological development": ["F80", "F89"],
    "Behavioural & emotional disorders of childhood/adolescence": ["F90", "F98"],
    "Unspecified mental disorder": ["F99", "F99"],
    "Inflammatory diseases of the CNS": ["G00", "G09"],
    "Systemic atrophies affecting the CNS": ["G10", "G14"],
    "Extrapyramidal & movement disorders": ["G20", "G26"],
    "Other degenerative diseases of the nervous system": ["G30", "G32"],
    "Demyelinating diseases of the CNS": ["G35", "G37"],
    "Episodic & paroxysmal disorders": ["G40", "G47"],
    "Nerve, nerve root & plexus disorders": ["G50", "G59"],
    "Polyneuropathies & other disorders of the PNS": ["G60", "G64"],
    "Diseases of myoneural junction & muscle": ["G70", "G73"],
    "Cerebral palsy & other paralytic syndromes": ["G80", "G83"],
    "Other disorders of the nervous system": ["G90", "G99"],
    "Disorders of eyelid, lacrimal system & orbit": ["H00", "H06"],
    "Disorders of conjunctiva": ["H10", "H13"],
    "Disorders of sclera, cornea, iris & ciliary body": ["H15", "H22"],
    "Disorders of lens": ["H25", "H28"],
    "Disorders of choroid & retina": ["H30", "H36"],
    "Glaucoma": ["H40", "H42"],
    "Disorders of vitreous body & globe": ["H43", "H45"],
    "Disorders of optic nerve & visual pathways": ["H46", "H48"],
    "Disorders of ocular muscles, binocular movement, accommodation & refraction": ["H49", "H52"],
    "Visual disturbances & blindness": ["H53", "H54"],
    "Other disorders of eye & adnexa": ["H55", "H59"],
    "Diseases of external ear": ["H60", "H62"],
    "Diseases of middle ear & mastoid": ["H65", "H75"],
    "Diseases of inner ear": ["H80", "H83"],
    "Other disorders of ear": ["H90", "H95"],
    "Acute rheumatic fever": ["I00", "I02"],
    "Chronic rheumatic heart diseases": ["I05", "I09"],
    "Hypertensive diseases": ["I10", "I15"],
    "Ischaemic heart diseases": ["I20", "I25"],
    "Pulmonary heart disease & diseases of pulmonary circulation": ["I26", "I28"],
    "Other heart diseases": ["I30", "I52"],
    "Cerebrovascular diseases": ["I60", "I69"],
    "Diseases of arteries, arterioles & capillaries": ["I70", "I79"],
    "Diseases of veins, lymphatic vessels & lymph nodes": ["I80", "I89"],
    "Other disorders of circulatory system": ["I95", "I99"],
    "Acute upper respiratory infections": ["J00", "J06"],
    "Influenza & pneumonia": ["J09", "J18"],
    "Other acute lower respiratory infections": ["J20", "J22"],
    "Other diseases of upper respiratory tract": ["J30", "J39"],
    "Chronic lower respiratory diseases": ["J40", "J47"],
    "Lung diseases due to external agents": ["J60", "J70"],
    "Other respiratory diseases principally affecting the interstitium": ["J80", "J84"],
    "Suppurative & necrotic conditions of lower respiratory tract": ["J85", "J86"],
    "Other respiratory diseases": ["J90", "J99"],
    "Diseases of oral cavity, salivary glands & jaw": ["K00", "K14"],
    "Diseases of oesophagus, stomach & duodenum": ["K20", "K31"],
    "Diseases of appendix": ["K35", "K38"],
    "Hernia": ["K40", "K46"],
    "Noninfective enteritis & colitis": ["K50", "K52"],
    "Other diseases of intestines": ["K55", "K63"],
    "Diseases of peritoneum & omentum": ["K65", "K68"],
    "Diseases of liver": ["K70", "K77"],
    "Disorders of gallbladder, biliary tract & pancreas": ["K80", "K87"],
    "Other diseases of the digestive system": ["K90", "K95"],
    "Infections of the skin & subcutaneous tissue": ["L00", "L08"],
    "Bullous disorders": ["L10", "L14"],
    "Dermatitis & eczema": ["L20", "L30"],
    "Papulosquamous disorders": ["L40", "L45"],
    "Urticaria & erythema": ["L50", "L54"],
    "Radiation-related skin disorders": ["L55", "L59"],
    "Disorders of skin appendages": ["L60", "L75"],
    "Other skin disorders": ["L80", "L99"],
    "Infectious arthropathies": ["M00", "M03"],
    "Inflammatory polyarthropathies": ["M05", "M14"],
    "Systemic connective tissue disorders": ["M30", "M36"],
    "Deforming dorsopathies": ["M40", "M43"],
    "Spondylopathies": ["M45", "M49"],
    "Other dorsopathies": ["M50", "M54"],
    "Soft tissue disorders": ["M60", "M79"],
    "Osteopathies & chondropathies": ["M80", "M94"],
    "Other musculoskeletal disorders": ["M95", "M99"],
    "Glomerular diseases": ["N00", "N08"],
    "Renal tubulo-interstitial diseases": ["N10", "N16"],
    "Renal failure": ["N17", "N19"],
    "Urolithiasis": ["N20", "N23"],
    "Other urinary diseases": ["N25", "N39"],
    "Male genital disorders": ["N40", "N51"],
    "Breast disorders": ["N60", "N65"],
    "Inflammatory disorders of female pelvic organs": ["N70", "N77"],
    "Noninflammatory disorders of female genital tract": ["N80", "N99"],
    "Pregnancy with abortive outcome": ["O00", "O08"],
    "Oedema, proteinuria & hypertensive disorders in pregnancy": ["O10", "O16"],
    "Complications mainly related to pregnancy": ["O20", "O29"],
    "Complications mainly related to labour & delivery": ["O60", "O75"],
    "Delivery": ["O80", "O84"],
    "Complications of puerperium": ["O85", "O92"],
    "Other obstetric conditions": ["O94", "O99"],
    "Fetus & newborn affected by maternal factors": ["P00", "P04"],
    "Disorders related to gestation length & fetal growth": ["P05", "P08"],
    "Birth trauma": ["P10", "P15"],
    "Respiratory & cardiovascular disorders in the perinatal period": ["P20", "P29"],
    "Infections specific to the perinatal period": ["P35", "P39"],
    "Haemorrhagic & haematological disorders of fetus & newborn": ["P50", "P61"],
    "Digestive system disorders of fetus & newborn": ["P75", "P78"],
    "Other perinatal conditions": ["P80", "P96"],
    "Congenital malformations of the nervous system": ["Q00", "Q07"],
    "Congenital malformations of eye, ear, face & neck": ["Q10", "Q18"],
    "Congenital malformations of circulatory system": ["Q20", "Q28"],
    "Congenital malformations of respiratory system": ["Q30", "Q34"],
    "Cleft lip & palate": ["Q35", "Q37"],
    "Other congenital malformations of digestive system": ["Q38", "Q45"],
    "Congenital malformations of genital organs": ["Q50", "Q56"],
    "Congenital malformations of urinary system": ["Q60", "Q64"],
    "Congenital malformations & deformities of musculoskeletal system": ["Q65", "Q79"],
    "Other congenital malformations": ["Q80", "Q89"],
    "Chromosomal abnormalities": ["Q90", "Q99"],
    "Symptoms & signs involving the circulatory & respiratory systems": ["R00", "R09"],
    "Symptoms & signs involving the digestive system & abdomen": ["R10", "R19"],
    "Symptoms & signs involving the skin & subcutaneous tissue": ["R20", "R23"],
    "Symptoms & signs involving the nervous & musculoskeletal systems": ["R25", "R29"],
    "Symptoms & signs involving cognitive functions & awareness": ["R40", "R46"],
    "General symptoms & signs": ["R50", "R69"],
    "Abnormal findings in clinical and laboratory studies": ["R70", "R94"],
    "Ill-defined & unknown causes of mortality": ["R95", "R99"],
    "Injuries to the head": ["S00", "S09"],
    "Injuries to the neck": ["S10", "S19"],
    "Injuries to the thorax": ["S20", "S29"],
    "Injuries to the abdomen, lower back, lumbar spine & pelvis": ["S30", "S39"],
    "Injuries to the shoulder & upper arm": ["S40", "S49"],
    "Injuries to the elbow & forearm": ["S50", "S59"],
    "Injuries to the wrist & hand": ["S60", "S69"],
    "Injuries to the hip & thigh": ["S70", "S79"],
    "Injuries to the knee & lower leg": ["S80", "S89"],
    "Injuries to the ankle & foot": ["S90", "S99"],
    "Effects of foreign body entering through natural orifice": ["T15", "T19"],
    "Burns & corrosions": ["T20", "T32"],
    "Poisoning by drugs, medicaments & biological substances": ["T36", "T50"],
    "Poisoning by other substances & toxic effects": ["T51", "T65"],
    "Other & unspecified effects of external causes": ["T66", "T78"],
    "Complications of surgical & medical care": ["T80", "T88"],
    "Sequelae of injuries, poisoning & other external causes": ["T90", "T98"],
    "Transport accidents": ["V01", "V99"],
    "Falls": ["W00", "W19"],
    "Exposure to mechanical forces": ["W20", "W49"],
    "Accidental drowning & submersion": ["W65", "W74"],
    "Other accidental threats to breathing": ["W75", "W84"],
    "Exposure to smoke, fire & flames": ["X00", "X09"],
    "Contact with heat & hot substances": ["X10", "X19"],
    "Contact with venomous animals & plants": ["X20", "X29"],
    "Exposure to forces of nature": ["X30", "X39"],
    "Accidental poisoning by and exposure to noxious substances": ["X40", "X49"],
    "Intentional self-harm": ["X60", "X84"],
    "Assault": ["X85", "Y09"],
    "Event of undetermined intent": ["Y10", "Y34"],
    "Legal intervention & operations of war": ["Y35", "Y36"],
    "Complications of medical & surgical care": ["Y40", "Y84"],
    "Supplementary factors related to morbidity & mortality": ["Y85", "Y89"],
    "Persons encountering health services for examination & investigation": ["Z00", "Z13"],
    "Persons encountering health services for preventive care": ["Z20", "Z29"],
    "Persons with potential health hazards related to family & personal history": ["Z30", "Z39"],
    "Persons encountering health services related to reproduction": ["Z30", "Z39"],
    "Persons with potential health hazards related to socioeconomic & psychosocial circumstances": ["Z55", "Z65"],
    "Persons encountering health services for specific procedures": ["Z40", "Z54"],
    "Persons with potential health hazards related to communicable diseases": ["Z20", "Z29"],
    "Contact with health services for other reasons": ["Z70", "Z76"],
    "Other factors influencing health status and contact with health services": ["Z80", "Z99"]
}


codes_agrupations_high_level = dict()
codes_agrupations_high_level["infectious and parasitic diseases"] = ["A00","B99"]
codes_agrupations_high_level["neoplasms"] = ["C00","D48"]
codes_agrupations_high_level["diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"] = ["D50","D89"]
codes_agrupations_high_level["endocrine, nutritional and metabolic diseases"] = ["E00","E90"]
codes_agrupations_high_level["mental and behavioural disorders"] = ["F00","F99"]
codes_agrupations_high_level["diseases of the nervous system"] = ["G00","G99"]
codes_agrupations_high_level["diseases of the eye and adnexa"] = ["H00","H59"]
codes_agrupations_high_level["diseases of the ear and mastoid process"] = ["H60","H95"]
codes_agrupations_high_level["diseases of the circulatory system"] = ["I00","I99"]
codes_agrupations_high_level["diseases of the respiratory system"] = ["J00","J99"]
codes_agrupations_high_level["diseases of the digestive system"] = ["K00","K93"]
codes_agrupations_high_level["diseases of the skin and subcutaneous tissue"] = ["L00","L99"]
codes_agrupations_high_level["diseases of the musculoskeletal system and connective tissue"] = ["M00","M99"]
codes_agrupations_high_level["diseases of the genitourinary system"] = ["N00","N99"]
codes_agrupations_high_level["pregnancy, childbirth and the puerperium"] = ["O00","O99"]
codes_agrupations_high_level["certain conditions originating in the perinatal period"] = ["P00","P96"]
codes_agrupations_high_level["congenital malformations, deformations and chromosomal abnormalities"] = ["Q00","Q99"]
codes_agrupations_high_level["symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"] = ["R00","R99"]
codes_agrupations_high_level["injury, poisoning and certain other consequences of external causes"] = ["S00","T98"]
codes_agrupations_high_level["external causes of morbidity and mortality"] = ["V01","Y98"]  
codes_agrupations_high_level["factors influencing health status and contact with health services"] = ["Z00","Z99"]

def agurpation_func(df, agrupation_dict):
    new_df = pd.DataFrame()
    for key, value in agrupation_dict.items():
        num1 = value[0][1:]
        num2 = value[1][1:]
        letter1 = value[0][0]
        letter2 = value[1][0] 

        if letter1 == letter2:
            codes_list = [letter1 + str(i).zfill(2) for i in range(int(num1), int(num2)+1)]

        else:
            codes_list1 = [letter1 + str(i).zfill(2) for i in range(int(num1),100)]
            codes_list2 = [letter2 + str(i).zfill(2) for i in range(0, int(num2)+1)]
            codes_list = codes_list1 + codes_list2

        codes_cols = set(df.columns)
        codes_list = set(codes_list)
        filtered_codes_list = list(codes_cols & codes_list)


        new_df[key] = df[filtered_codes_list].sum(axis=1)

    return new_df


def map_ups(df, up_mapping):
    pass

def data_seasonality(df):
    pass



if __name__ == "__main__":


    # Example usage
    df = pd.read_csv('../data/date_2008-01-01_longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv')

    print("Original DataFrame:")
    print(df)

    new_df = agurpation_func(df, codes_agrupations_low_level)

    print("\nDataFrame after aggregation:")
    print(new_df)