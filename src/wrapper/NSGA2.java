package wrapper;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import jmetal.problems.ProblemFactory;
import jmetal.problems.ProblemAlber;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.metaheuristics.nsgaII.*;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

/** 
 * Class to configure and execute the NSGA-II algorithm.  
 *     
 * Besides the classic NSGA-II, a steady-state version (ssNSGAII) is also
 * included (See: J.J. Durillo, A.J. Nebro, F. Luna and E. Alba 
 *                  "On the Effect of the Steady-State Selection Scheme in 
 *                  Multi-Objective Genetic Algorithms"
 *                  5th International Conference, EMO 2009, pp: 183-197. 
 *                  April 2009)
 */ 

public class NSGA2 {
  public static Logger      logger_ ;      // Logger object
  public static FileHandler fileHandler_ ; // FileHandler object

  private int popSize,evals,instances,fitness;
  private double seed, crossover, mutation;
  private myDataset train;

  public NSGA2() {
	   //nada
  }
  
  public NSGA2(myDataset train, long seed, int popSize, int evals, double crossover, double mutation, int instances, int fitness) {
	  Long prueba = new Long(seed);
	  String cadena = prueba.toString();
	  int zeros = cadena.length();
	  int divisor = 1;
	  for (int i = 0; i < zeros; i++){
		  divisor *= 10;
	  }
	  this.seed = 1.0*seed/divisor;
	  this.popSize = popSize;
	  this.evals = evals;
	  this.crossover = crossover;
	  this.mutation = mutation;
	  this.train = train;
	  this.instances = instances;
	  this.fitness = fitness;
 }
  
  /**
   * @param args Command line arguments.
   * @throws JMException 
   * @throws IOException 
   * @throws SecurityException 
   * Usage: three options
   *      - jmetal.metaheuristics.nsgaII.NSGAII_main
   *      - jmetal.metaheuristics.nsgaII.NSGAII_main problemName
   *      - jmetal.metaheuristics.nsgaII.NSGAII_main problemName paretoFrontFile
   */
  public void execute() throws    JMException, 
                                  SecurityException, 
                                  IOException, 
                                  ClassNotFoundException {
    Problem   problem   ; // The problem to solve
    Algorithm algorithm ; // The algorithm to use
    Operator  crossover ; // Crossover operator
    Operator  mutation  ; // Mutation operator
    Operator  selection ; // Selection operator
    
    HashMap  parameters ; // Operator parameters
    
    QualityIndicator indicators ; // Object to get quality indicators

    // Logger object and file to store log messages
    logger_      = Configuration.logger_ ;
    fileHandler_ = new FileHandler(Wrapper.header+".log"); 
    logger_.addHandler(fileHandler_) ;
        
    indicators = null ;
    PseudoRandom.setSeed(seed);
    
    problem = new ProblemC45("Binary",train,instances,fitness);
    
    algorithm = new NSGAII(problem);
    //algorithm = new ssNSGAII(problem);

    // Algorithm parameters
    algorithm.setInputParameter("populationSize",popSize);
    algorithm.setInputParameter("maxEvaluations",evals);

    // Mutation and Crossover for Real codification 
    
    parameters = new HashMap() ;
    /*
    parameters.put("probability", 0.9) ;
    parameters.put("distributionIndex", 20.0) ;
    */
    //parameters.put("realCrossoverProbability", this.crossover) ;
    parameters.put("Xprobability", this.crossover) ;
    parameters.put("distributionIndex", 20.0) ;
    crossover = CrossoverFactory.getCrossoverOperator("HUXCrossover", parameters);                   

    parameters = new HashMap() ;
    /*
    parameters.put("probability", 1.0/problem.getNumberOfVariables()) ;
    parameters.put("distributionIndex", 20.0) ;
    */
    //parameters.put("realMutationProbability", 1.0/problem.getNumberOfVariables()) ;
    parameters.put("MutProbability", 1.0/problem.getNumberOfVariables()) ;
    parameters.put("distributionIndex", 20.0) ;
    mutation = MutationFactory.getMutationOperator("BitFlipMutation", parameters);                    

    // Selection Operator 
    parameters = null ;
    selection = SelectionFactory.getSelectionOperator("BinaryTournament2", parameters) ;                           

    // Add the operators to the algorithm
    algorithm.addOperator("crossover",crossover);
    algorithm.addOperator("mutation",mutation);
    algorithm.addOperator("selection",selection);

    // Add the indicator object to the algorithm
    algorithm.setInputParameter("indicators", indicators) ;
    
    // Execute the Algorithm
    long initTime = System.currentTimeMillis();
    System.err.println("Starting!");
    SolutionSet population = algorithm.execute();
    System.err.println("Executed!");
    long estimatedTime = System.currentTimeMillis() - initTime;
    
    // Result messages 
    logger_.info("Total execution time: "+estimatedTime + "ms");
    logger_.info("Variables values have been writen to file VAR");
    population.printVariablesToFile(Wrapper.header+".var");    
    logger_.info("Objectives values have been writen to file FUN");
    population.printObjectivesToFile(Wrapper.header+".fun");
    
    if (indicators != null) {
      logger_.info("Quality indicators") ;
      logger_.info("Hypervolume: " + indicators.getHypervolume(population)) ;
      logger_.info("GD         : " + indicators.getGD(population)) ;
      logger_.info("IGD        : " + indicators.getIGD(population)) ;
      logger_.info("Spread     : " + indicators.getSpread(population)) ;
      logger_.info("Epsilon    : " + indicators.getEpsilon(population)) ;  
     
      int evaluations = ((Integer)algorithm.getOutputParameter("evaluations")).intValue();
      logger_.info("Speed      : " + evaluations + " evaluations") ;      
    } // if
  } //main
  
} // NSGAII_main
