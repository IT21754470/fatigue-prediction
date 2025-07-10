"""
Utility functions and constants for swimming recommendations
"""

# Recommendation mappings for grshort codes
RECOMMENDATIONS = {
    "MC": "maintain current training load with emphasis on backstroke-specific core work. Add 4×100m descending pace sets with 45s rest.",
    "MCTAW": "Maintain current training approach with balanced intensity. Focus on maintaining proper stroke mechanics throughout sets.",
    "MCT": "Maintain current training approach with balanced focus on technique and endurance.",
    "FOT": "Focus on technique rather than volume. Add specific drills targeting your weakest phase of the stroke.",
    "RDC": "Reduce drill complexity but maintain focus. Concentrate on mastering 2-3 fundamental drills rather than variety.",
    "FO": "Focus on improving underwater phase and breakout technique. Add dolphin kick drills on your back.",
    "ITFS": "Incorporate 4×200m technique-focused sets with paddles. Emphasize distance per stroke rather than speed",
    "BDW": "Balance drill work between technique refinement and endurance development with equal focus",

    "ID": "Incorporate drill-swim sets (25m drill/25m swim) focusing on one technical element at a time.",
  
    "ACS": "Add core stability exercises that specifically support the body positions required in your drill sequences.",
    "HER": "Ensure 48-hour recovery periods between intensive drill sessions to allow for proper motor pattern development",
    "MCTVW": "Maintain current training volume while adding one race-pace set per week (6×50m at race pace with 45s rest)",
    "EA": "Ensure adequate recovery between backstroke sessions with at least 48 hours between specialized sets.",
    "ABS":"Add breaststroke-specific core work focusing on hip flexibility and strength",
    "ABSS":"Add butterfly-specific shoulder stability exercises to prevent injury.",
    "EAR": "Ensure adequate recovery between breaststroke sessions with at least 48 hours between specialized sets.",
     "EARP": "Ensure adequate recovery with 25-30g protein within 30 minutes post-workout and increase complex carbohydrates (60-80g) for energy restoration",
    "EAV": "Ensure adequate vitamin D intake (1000-2000 IU daily) and 20g protein within 30 minutes post-workout.",
    "EPN": "Ensure proper nutrition with emphasis on anti-inflammatory foods and adequate hydration.",
    "FOTR": "Focus on technique refinement rather than volume. Add specific drills targeting body position and timing.",
    "IDS": "Increase drill-specific focus on technical elements. Add 8×50m drill sets targeting your weakest stroke components.",
    "IUDK": "Incorporate underwater dolphin kicks after turns to improve transitions.",
    "SWM": "Supplement with 300-400mg magnesium and 20-25g protein within 30 minutes post-workout to support muscle recovery.",
    
    "IPWC": "Include 20g protein with 60-80g carbohydrates post-workout for optimal recovery.",
    "APS": "Add proprioceptive exercises with resistance bands to reinforce proper movement patterns outside of water.",
    "SWMF": "Supplement with omega-3 fatty acids (1-2g daily) to support joint health and reduce inflammation.",
    
    "MCD": "Maintain current drill volume while adding one technique-focused session weekly with coach feedback.",
    "WOM": "Work on maintaining proper body alignment during all drill sequences. Use video feedback to identify inefficiencies.",
    "EARB": "Ensure adequate recovery between butterfly sessions with at least 48 hours between specialized sets.",
    "APE": "Add proprioceptive exercises with resistance bands to reinforce proper movement patterns outside of water.",
    "ISS": "Incorporate stroke-specific drills that translate directly to your primary competitive strokes",
   
    
    "IHWE": "Increase hydration with electrolytes (600-800mg sodium) during workouts and 25g protein post-training.",
    "EPH": "Ensure proper hydration with electrolyte-enhanced fluids (500-700mg sodium) during longer drill sessions.",
    "ADC": "Adjust distance and conditioning approach. Modify endurance training strategy.",
    "EPNW": "Ensure proper nutrition with emphasis on anti-inflammatory foods and adequate hydration",
    "BHTD": "Balance hard training days with recovery sessions. Add stroke count drills to improve efficiency.",
    "GPW": "Great progress with drills! Add one challenging drill-to-swimming transition set weekly (4×100m with 25m drill/75m swim).",
    "EPM": "Excellent progress! Maintain current training while adding race-specific pacing work (4×50m at race pace with full recovery).",
    "SARW": "Schedule a recovery week with 30% reduced volume but maintain some quality work.",
    "RTVB": "Reduce training volume by 20-30%. Focus on recovery and technique work until fatigue decreases",
    "PAR": "Prioritize active recovery (easy swims, technique drills) and ensure adequate rest between sessions.",
    "RTVBF": "Recovery training with varied intensity and increased frequency. More recovery-focused sessions.",
    "ETVB": "Ensure adequate vitamin D intake (1000-2000 IU daily) and 20g protein within 30 minutes post-workout",
    "IRIA": "Increase rest intervals and incorporate more low-intensity, skill-focused sets.",
    "FOTER": "Focus on technique efficiency rather than volume. Add video analysis sessions.",
    "IAIF": "Increase anti-inflammatory foods in your diet and ensure adequate sleep (8-9 hours).",
    "EPHW": "Ensure proper hydration with electrolytes and increase protein intake to 1.8-2g per kg bodyweight.",
    "ISRT": "Implement stress reduction techniques such as meditation or yoga to enhance recovery",
    "SWMA": "Supplement with magnesium (300-400mg) and zinc (15-30mg) to support muscle recovery.",
    "IAR": "Improve aerobic capacity and recovery. Balance endurance development with adequate rest.",
    "KAS": "Keep a streamlined body position, coordinate arm pull and leg kick timing, and avoid overextending arms"
}

def get_recommendation_text(grshort_code):
    """Get recommendation text for a grshort code"""
    return RECOMMENDATIONS.get(
        grshort_code, 
        "General training recommendation based on your current metrics and performance."
    )

def format_prediction_response(swimmer_id, stroke_type, improvement, fatigue_level, 
                             grshort_code, confidence, top_predictions):
    """Format the prediction response"""
    return {
        "swimmer_id": swimmer_id,
        "stroke_type": stroke_type,
        "predicted_improvement": improvement,
        "fatigue_level": fatigue_level,
        "predicted_grshort": grshort_code,
        "confidence": round(confidence, 3),
        "full_recommendation": get_recommendation_text(grshort_code),
        "top_predictions": [
            {"grshort": code, "confidence": round(prob, 3)} 
            for code, prob in top_predictions
        ]
    }