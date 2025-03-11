#Converting to CSV
import scipy.io
import pandas as pd

# Load FADHDRedTrans.mat
fadhd_data = scipy.io.loadmat('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\ADHD Data Reduction\FADHDRedTrans.mat')
# Extract the data
fadhd_df = pd.DataFrame(fadhd_data['FADHD'])

# Save to CSV
fadhd_df.to_csv('FADHDRedTrans.csv', index=False)

# Load FCRedTrans.mat
fc_data = scipy.io.loadmat('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\ADHD Data Reduction\FCRedTrans.mat')
# Extract the data
fc_df = pd.DataFrame(fc_data['FC'])

# Save to CSV
fc_df.to_csv('FCRedTrans.csv', index=False)

# Load MADHDRedTrans.mat
madhd_data = scipy.io.loadmat('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\ADHD Data Reduction\MADHDRedTrans.mat')
# Extract the data
madhd_df = pd.DataFrame(madhd_data['MADHD'])

# Save to CSV
madhd_df.to_csv('MADHDRedTrans.csv', index=False)

# Load MCRedTrans.mat
mc_data = scipy.io.loadmat('F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\ADHD Data Reduction\MCRedTrans.mat')
# Extract the data
mc_df = pd.DataFrame(mc_data['MC'])

# Save to CSV
mc_df.to_csv('MCRedTrans.csv', index=False)
