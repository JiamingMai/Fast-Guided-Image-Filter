import java.util.ArrayList;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class FastGuidedFilter {	
	ArrayList<Mat> Ichannels;
	ArrayList<Mat> Isubchannels;
	int Idepth;
	int r;
	double eps;
	double s;
	double r_sub;
	Mat mean_I_r;
	Mat mean_I_g;
	Mat mean_I_b;
	Mat invrr;
	Mat invrg;
	Mat invrb;
	Mat invgg;
	Mat invgb;
	Mat invbb;
	
	public FastGuidedFilter(){
		Ichannels = new ArrayList<Mat>();
		Isubchannels = new ArrayList<Mat>();
		invrr = new Mat();
		invrg = new Mat();
		invrb = new Mat();
		invgg = new Mat();
		invgb = new Mat();
		invbb = new Mat();
	}
	
    public static Mat boxfilter(Mat I, int r){
    	Mat result = new Mat();
        Imgproc.blur(I, result, new Size(r, r));
    	return result;
    }
    
    public static Mat convertTo(Mat mat, int depth){
    	if(mat.depth() == depth){
    		return mat;
    	}
    	Mat result = new Mat();
    	mat.convertTo(result, depth);
    	return result;
    }
    
    public Mat filterSingleChannel(Mat p, double s){
    	Mat p_sub = new Mat();
    	Imgproc.resize(p, p_sub, new Size(p.cols()/s, p.rows()/s), 0.0, 0.0, Imgproc.INTER_NEAREST);
    	
    	Mat mean_p = boxfilter(p_sub, (int)r_sub);
    	
    	Mat mean_Ip_r = boxfilter(Isubchannels.get(0).mul(p_sub), (int)r_sub);
    	Mat mean_Ip_g = boxfilter(Isubchannels.get(1).mul(p_sub), (int)r_sub);
    	Mat mean_Ip_b = boxfilter(Isubchannels.get(2).mul(p_sub), (int)r_sub);
    	
    	// convariance of (I, p) in each local patch
    	Mat cov_Ip_r = new Mat();
    	Mat cov_Ip_g = new Mat();
    	Mat cov_Ip_b = new Mat();
    	Core.subtract(mean_Ip_r, mean_I_r.mul(mean_p), cov_Ip_r);
    	Core.subtract(mean_Ip_g, mean_I_g.mul(mean_p), cov_Ip_g);
    	Core.subtract(mean_Ip_b, mean_I_b.mul(mean_p), cov_Ip_b);
    	
    	Mat temp1 = new Mat();
    	Mat a_r = new Mat();
    	Mat a_g = new Mat();
    	Mat a_b = new Mat();
    	Core.add(invrr.mul(cov_Ip_r), invrg.mul(cov_Ip_g), temp1);
    	Core.add(temp1, invrb.mul(cov_Ip_b), a_r);
    	Core.add(invrg.mul(cov_Ip_r), invgg.mul(cov_Ip_g), temp1);
    	Core.add(temp1, invgb.mul(cov_Ip_b), a_g);
    	Core.add(invrb.mul(cov_Ip_r), invgb.mul(cov_Ip_g), temp1);
    	Core.add(temp1, invbb.mul(cov_Ip_b), a_b);    
    	
    	Mat b = new Mat();
        Core.subtract(mean_p, a_r.mul(mean_I_r), b);
        Core.subtract(b, a_g.mul(mean_I_g), b);
        Core.subtract(b, a_b.mul(mean_I_b), b);
    	
        Mat mean_a_r = boxfilter(a_r, (int)r_sub);
        Mat mean_a_g = boxfilter(a_g, (int)r_sub);
        Mat mean_a_b = boxfilter(a_b, (int)r_sub);
        Mat mean_b = boxfilter(b, (int)r_sub);
        
        Imgproc.resize(mean_a_r, mean_a_r, new Size(Ichannels.get(0).cols(), Ichannels.get(0).rows()), 0.0, 0.0, Imgproc.INTER_LINEAR);
        Imgproc.resize(mean_a_g, mean_a_g, new Size(Ichannels.get(0).cols(), Ichannels.get(0).rows()), 0.0, 0.0, Imgproc.INTER_LINEAR);
        Imgproc.resize(mean_a_b, mean_a_b, new Size(Ichannels.get(0).cols(), Ichannels.get(0).rows()), 0.0, 0.0, Imgproc.INTER_LINEAR);
        Imgproc.resize(mean_b, mean_b, new Size(Ichannels.get(0).cols(), Ichannels.get(0).rows()), 0.0, 0.0, Imgproc.INTER_LINEAR);
        
        Mat result = new Mat();
        Core.add(mean_a_r.mul(Ichannels.get(0)), mean_a_g.mul(Ichannels.get(1)), temp1);
        Core.add(temp1, mean_a_b.mul(Ichannels.get(2)), temp1);
        Core.add(temp1, mean_b, result);
        return result;
    }
    
    public Mat filter(Mat origI, Mat p, int r, double eps, double s, int depth){
    	Mat I;
    	if(origI.depth() == CvType.CV_32F || origI.depth() == CvType.CV_64F){
    		I = origI.clone();    	    
    	}
    	else{
    		I = convertTo(origI, CvType.CV_32F);
    	}
    	Idepth = I.depth();
    	Core.split(I, Ichannels);
    	Mat I_sub = new Mat();
    	Imgproc.resize(I, I_sub, new Size(I.cols()/s, I.rows()/s), 0.0, 0.0, Imgproc.INTER_NEAREST);
    	Core.split(I_sub, Isubchannels);
    	r_sub = r / s;
    	mean_I_r = boxfilter(Isubchannels.get(0), (int)r_sub);
    	mean_I_g = boxfilter(Isubchannels.get(1), (int)r_sub);
    	mean_I_b = boxfilter(Isubchannels.get(2), (int)r_sub);
    	
    	// variance of I in each local patch: the matrix Sigma in Eqn (14).
        // Note the variance in each local patch is a 3x3 symmetric matrix:
        //           rr, rg, rb
        //   Sigma = rg, gg, gb
        //           rb, gb, bb    	
    	Mat var_I_rr = new Mat();
    	Mat var_I_rg = new Mat(); 
    	Mat var_I_rb = new Mat(); 
    	Mat var_I_gg = new Mat(); 
    	Mat var_I_gb = new Mat(); 
    	Mat var_I_bb = new Mat();
    	Mat temp1 = new Mat();
    	
    	Core.subtract(boxfilter(Isubchannels.get(0).mul(Isubchannels.get(0)), (int)r_sub), mean_I_r.mul(mean_I_r), temp1);
    	Core.add(temp1, new Scalar(eps), var_I_rr);
    	Core.subtract(boxfilter(Isubchannels.get(0).mul(Isubchannels.get(1)), (int)r_sub), mean_I_r.mul(mean_I_g), var_I_rg);
    	Core.subtract(boxfilter(Isubchannels.get(0).mul(Isubchannels.get(2)), (int)r_sub), mean_I_r.mul(mean_I_b), var_I_rb);
    	Core.subtract(boxfilter(Isubchannels.get(1).mul(Isubchannels.get(1)), (int)r_sub), mean_I_g.mul(mean_I_g), temp1);
    	Core.add(temp1, new Scalar(eps), var_I_gg);
    	Core.subtract(boxfilter(Isubchannels.get(1).mul(Isubchannels.get(2)), (int)r_sub), mean_I_g.mul(mean_I_b), var_I_gb);
    	Core.subtract(boxfilter(Isubchannels.get(2).mul(Isubchannels.get(2)), (int)r_sub), mean_I_b.mul(mean_I_b), temp1);
    	Core.add(temp1, new Scalar(eps), var_I_bb);
    	
    	// Inverse of Sigma + eps * I
    	Core.subtract(var_I_gg.mul(var_I_bb), var_I_gb.mul(var_I_gb), invrr);
    	Core.subtract(var_I_gb.mul(var_I_rb), var_I_rg.mul(var_I_bb), invrg);
    	Core.subtract(var_I_rg.mul(var_I_gb), var_I_gg.mul(var_I_rb), invrb);
    	Core.subtract(var_I_rr.mul(var_I_bb), var_I_rb.mul(var_I_rb), invgg);
    	Core.subtract(var_I_rb.mul(var_I_rg), var_I_rr.mul(var_I_gb), invgb);
    	Core.subtract(var_I_rr.mul(var_I_gg), var_I_rg.mul(var_I_rg), invbb);
    	
    	Mat covDet = new Mat();
    	Core.add(invrr.mul(var_I_rr), invrg.mul(var_I_rg), temp1);
    	Core.add(temp1, invrb.mul(var_I_rb), covDet);
    	
    	Core.divide(invrr, covDet, invrr);
    	Core.divide(invrg, covDet, invrg);
    	Core.divide(invrb, covDet, invrb);
    	Core.divide(invgg, covDet, invgg);
    	Core.divide(invgb, covDet, invgb);
    	Core.divide(invbb, covDet, invbb);
    	
    	Mat p2 = convertTo(p, Idepth);
    	Mat result = new Mat();
    	if(p.channels() == 1){
    		result = filterSingleChannel(p2, s);
    	}else{
    		ArrayList<Mat> pc = new ArrayList<Mat>();
    		Core.split(p2, pc);
    		for(int i = 0; i < pc.size(); i++){
    			pc.set(i, filterSingleChannel(pc.get(i), s));
    		}
    		Core.merge(pc, result);
    	}
    	return convertTo(result, depth == -1 ? p.depth() : depth);
    }
    
}
