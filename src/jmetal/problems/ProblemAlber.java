package jmetal.problems;

import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.Variable;
import jmetal.encodings.solutionType.ArrayRealAndBinarySolutionType;
import jmetal.encodings.variable.ArrayReal;
import jmetal.encodings.variable.Binary;
import jmetal.util.JMException;
import jmetal.util.wrapper.XReal;

/** 
 * Class representing problem ProblemAlber
 */
public class ProblemAlber extends Problem {
    
	private int boolVariables_ ;
	private int realVariables_ ;
	
 /**
  * Constructor.
  * Creates default instance of problem ZDT3 (30 decision variables.
  * @param solutionType The solution type must "Real", "BinaryReal, and "ArrayReal". 
  */    
  public ProblemAlber(String solutionType) throws ClassNotFoundException {
    this(solutionType, 30, 10); // 30 variables by default
  } // ZDT3
  
  
 /** 
  * Constructor.
  * Creates a instance of ProblemAlber problem.
  * @param numberOfVariables Number of variables.
  * @param solutionType The solution type must "BinaryReal", "BinaryReal, and "ArrayReal". 
  */    
  public ProblemAlber(String solutionType, int numberOfLabels, int numberOfRules) {
    numberOfVariables_  = numberOfLabels + numberOfRules;
	boolVariables_ =  numberOfRules ;
	length_ = new int[2];
	length_[0] = numberOfLabels;
	length_[1] = numberOfRules;
	realVariables_ = numberOfLabels ;
    numberOfObjectives_ =  2;
    numberOfConstraints_=  0;
    problemName_        = "ProblemAlber";
        
    upperLimit_ = new double[numberOfVariables_];
    lowerLimit_ = new double[numberOfVariables_];
        
    for (int var = 0; var < numberOfVariables_; var++){ //el binario entre 0 y 1 y el real tb
      lowerLimit_[var] = 0.0;
      upperLimit_[var] = 1.0;
    } // for
        
    if (solutionType.compareTo("ArrayRealAndBinary") == 0){
    	solutionType_ = new ArrayRealAndBinarySolutionType(this,numberOfLabels,numberOfRules) ;
    }else {
    	System.out.println("Error: solution type " + solutionType + " invalid") ;
    	System.exit(-1) ;
    }
  } //ZDT3
      
  
  /** 
  * Evaluates a solution 
  * @param solution The solution to evaluate
   * @throws JMException 
  */    
  public void evaluate(Solution solution) throws JMException {
	//XReal x = new XReal(solution) ;
	Variable[] variables = solution.getDecisionVariables();
	
	ArrayReal solReal = (ArrayReal)variables[0];
	Binary solBinario = (Binary)variables[1];
	//variables[0] = ArrayReal
	//variables[1] = Binary

	double [] fx = new double[numberOfObjectives_] ; // function values     
	   
    fx[0] = 0.0 ;
    //System.err.print("Soluciones Reales: ");
    for (int var = 0; var < solReal.getLength() ; var++) {        
      fx[0] += solReal.getValue(var) < 0.5 ? 1 : 0; //
      //System.err.print(", "+solReal.getValue(var));
    } // for
        
    fx[1] = 0.0 ;
    //System.err.println("\nF[0]: "+fx[0]);
    //System.err.print("Soluciones Booleanas: ");
    for (int var = 0; var < solBinario.getNumberOfBits(); var++) {        
      fx[1] += solBinario.getIth(var) ? 1 : 0;
      //System.err.print(", "+solBinario.getIth(var));
    } // for
    //System.err.println("\nF[1]: "+fx[1]);
        
    solution.setObjective(0, fx[0]);
    solution.setObjective(1, fx[1]);
  } //evaluate
    
  /**
   * Returns the value of the ZDT2 function G.
   * @param  x Solution
   * @throws JMException 
   */    
  private double evalG(XReal x) throws JMException {
    double g = 0.0;        
    for (int i = 1; i < x.getNumberOfDecisionVariables();i++)
      g += x.getValue(i);
    double constant = (9.0 / (numberOfVariables_-1));
    g = constant * g;
    g = g + 1.0;
    return g;
  } //evalG
   
  /**
  * Returns the value of the ZDT3 function H.
  * @param f First argument of the function H.
  * @param g Second argument of the function H.
  */
  public double evalH(double f, double g) {
    double h = 0.0;
    h = 1.0 - java.lang.Math.sqrt(f/g) 
        - (f/g)*java.lang.Math.sin(10.0*java.lang.Math.PI*f);
    return h;        
  } //evalH
  
  public int getNVariablesReal(){
	  return this.realVariables_;
  }
  
  public int getNVariablesBool(){
	  return this.boolVariables_;
  }
  
} // ZDT3
