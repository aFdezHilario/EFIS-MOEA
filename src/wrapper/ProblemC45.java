package wrapper;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.Variable;
import jmetal.encodings.solutionType.ArrayRealAndBinarySolutionType;
import jmetal.encodings.solutionType.BinaryRealSolutionType;
import jmetal.encodings.solutionType.BinarySolutionType;
import jmetal.encodings.variable.ArrayReal;
import jmetal.encodings.variable.Binary;
import jmetal.util.JMException;
import jmetal.util.wrapper.XReal;
import keel.Algorithms.Decision_Trees.C45.C45;

/** 
 * Class representing problem ProblemAlber
 */
public class ProblemC45 extends Problem {

	private int features_ ;
	private int instances_ ;
	private int fitness;
	private myDataset train;
	private boolean weighting; //whether to use instance weights

	/**
	 * Constructor.
	 * Creates default instance of problem 
	 * @param solutionType The solution type must "Real", "BinaryReal, and "ArrayReal". 
	 */    
	public ProblemC45(String solutionType) throws ClassNotFoundException {
		this(solutionType, null,0,0,false);
	} 


	/** 
	 * Constructor.
	 * Creates a instance of ProblemC45.
	 * @param numberOfVariables Number of variables.
	 * @param solutionType The solution type must "BinaryReal", "BinaryReal, and "ArrayReal". 
	 */    
	public ProblemC45(String solutionType, myDataset train, int instances, int fitness, boolean weighting) {
		this.train = train;
		this.fitness = fitness;
		this.features_ = train.getnInputs();
		this.instances_ = train.getnData();
		this.weighting = weighting;
		if (instances == Wrapper.MAJ){
			this.instances_ = train.getMajority();
		}
		numberOfVariables_ = 2; 
		
		numberOfObjectives_ =  2;  // AUC + IS
		numberOfConstraints_=  0;
		problemName_        = "ProblemFS-IS";

		solutionType_ = new BinarySolutionType(this) ;
		
		length_ = new int[numberOfVariables_];
		length_ [0] = features_;
		length_ [1] = instances_;
		
		if (solutionType.compareTo("Binary") == 0){
			solutionType_ = new BinarySolutionType(this);
		}else {
			System.out.println("Error: solution type " + solutionType + " invalid") ;
			System.exit(-1) ;
		}
	}


	/** 
	 * Evaluates a solution 
	 * @param solution The solution to evaluate
	 * @throws JMException 
	 */    
	public void evaluate(Solution solution) throws JMException {
		Variable[] solutions = solution.getDecisionVariables();

		Binary solFS = (Binary)solutions[0];
		Binary solIS = (Binary)solutions[1];

		double [] fx = new double[numberOfObjectives_] ; // function values
		int vars = 0;
		boolean [] fs = new boolean[features_];
		boolean [] is = new boolean[instances_];
		for (int i = 0; i < fs.length; i++){
			fs[i] = solFS.getIth(i);
			if(fs[i]){
				vars++;
			}
		}
		for (int i = 0; i < is.length; i++){
			is[i] = solIS.getIth(i);
			if(is[i]){
				fx[1]++;
			}
		}

		if((vars == 0)||(fx[1] == 0)){
			fx[0] = 0;
		}else{
			try{
				C45 model = new C45(train,fs,is,weighting);
				if (fitness == Wrapper.aucVal){
					model.evaluateTest();
					fx[0] = model.getAUC();
				}else if (fitness == Wrapper.aucTrain){
					model.evaluateModel();
					fx[0] = model.getAUCModel();
				}
				else{
					model.evaluateTest();
					fx[0] = model.getGM();
				}
			}catch(Exception e){
				e.printStackTrace(System.err);;
				System.exit(-1);
			}
		}
		/******************************/
		/* CHANGE OBJECTIVE FUNCTIONS */
		/******************************/

		
		// Many more combinations should be studied
		/********************************/
		if(fx[1] == 0) fx[0] = -Double.MAX_VALUE;
		if(fx[0] == 0) fx[0] = -Double.MAX_VALUE;
		solution.setObjective(0, -1.0*fx[0]);
		solution.setObjective(1, 1.0*fx[1]);

	} //evaluate
	
	/**
	 * It returns the number of selected instances
	 * @param geneR the gene codification
	 * @return the number of selected instances
	 */
	public int getnSelected(int [] geneR) {
		int i, count;

		count = 0;
		for (i=0; i < geneR.length; i++) {
			if (geneR[i] > 0)  count++;
		}
		return (count);
	}

	
} // ProblemC45
