import numpy as np
import pandas as pd


def make_weights(args,data):
    data.reset_index(inplace=True)
    if args.task=="Stage":
        num_classes=args.n_classes
        data["temp_label"]=data['stage_value_map'].copy()
    elif args.task == "CancerCls":
        num_classes=args.n_classes
        data["temp_label"]=data['cancer_label'].copy()
    else:
        label_dict = {}
        key_index = 0
        for i in range(args.n_classes):
            for c in [0, 1]:
                label_dict.update({(i, c): key_index})
                key_index += 1
        data["temp_label"]=np.nan
        for i in range(data.shape[0]):
            if args.task=="OS":
                censorship_var="os_censorship"
                key=data.loc[i,"os_disc_labels"]
            elif args.task=="PFS":
                censorship_var="pfs_censorship"
                key = data.loc[i, "pfs_disc_labels"]
            censorship = data.loc[i, censorship_var]
            key = (key, int(censorship))
            data.loc[i, 'temp_label'] = label_dict[key]
        num_classes=args.n_classes*len([0,1])

    slide_cls_ids = [[] for _ in range(num_classes)]
    for i in range(num_classes):
        slide_cls_ids[i] = np.where(data['temp_label'] == i)[0]
    N = float(data.shape[0])
    weight_per_class=[]
    for c in range(len(slide_cls_ids)):
        if len(slide_cls_ids[c])>0:
            weight_per_class.append(N / len(slide_cls_ids[c]))
        else:
            weight_per_class.append(0)
    # weight_per_class = [N / len(slide_cls_ids[c]) for c in range(len(slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(data.shape[0]):
        y = int(data.loc[idx, 'temp_label'])
        weight[idx] = weight_per_class[y]
    return weight



class SurvivalDatasetFactory:

    def __init__(self,
        label_file,
        n_bins,
        label_col,censorship_var,
        eps=1e-6
        ):
        self.label_file = label_file
        self.n_bins = n_bins
        self.label_col = label_col#Label Column Names
        self.censorship_var = censorship_var

        self._setup_metadata_and_labels(eps)

    def _setup_metadata_and_labels(self, eps):
        r"""
        Process the metadata required to run the experiment. Clean the data. Set up patient dicts to store slide ids per patient.
        Get label dict.

        Args:
            - self
            - eps : Float

        Returns:
            - None

        """

        # ---> read labels
        self.label_data = self.label_file# pd.read_csv(self.label_file, low_memory=False)

        # ---> minor clean-up of the labels
        uncensored_df = self._clean_label_data()

        # ---> create discrete labels
        self._discretize_survival_months(eps, uncensored_df)




    def _clean_label_data(self):
        r"""
        Clean the metadata. For breast, only consider the IDC subtype.

        Args:
            - self

        Returns:
            - None

        """

        # if "IDC" in self.label_data['oncotree_code']:  # must be BRCA (and if so, use only IDCs)
        #     self.label_data = self.label_data[self.label_data['oncotree_code'] == 'IDC']

        self.patients_df = self.label_data.drop_duplicates(['case_id']).copy()
        uncensored_df = self.patients_df[self.patients_df[self.censorship_var] < 1]

        return uncensored_df

    def _discretize_survival_months(self, eps, uncensored_df):
        r"""
        This is where we convert the regression survival problem into a classification problem. We bin all survival times into
        quartiles and assign labels to patient based on these bins.

        Args:
            - self
            - eps : Float
            - uncensored_df : pd.DataFrame

        Returns:
            - None

        """
        # cut the data into self.n_bins (4= quantiles)
        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = self.label_data[self.label_col].max() + eps
        q_bins[0] = self.label_data[self.label_col].min() - eps

        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        self.disc_labels, self.q_bins = pd.cut(self.patients_df[self.label_col], bins=q_bins, retbins=True, labels=False,
                                     right=False, include_lowest=True)
        # self.patients_df.insert(2, 'disc_label', disc_labels.values.astype(int))
        # self.bins = q_bins
        # self.label_data = self.patients_df
    def get_target(self):
        return self.disc_labels, self.q_bins

    # def _get_patient_data(self):
    #     r"""
    #     Final patient data is just the clinical metadata + label for the patient
    #
    #     Args:
    #         - self
    #
    #     Returns:
    #         - None
    #
    #     """
    #     patients_df = self.label_data[~self.label_data.index.duplicated(keep='first')]#remove duplication
    #     patient_data = {'case_id': patients_df["case_id"].values,
    #                     'label': patients_df['label'].values}  # only setting the final data to self
    #     self.patient_data = patient_data

    # def _get_label_dict(self):
    #     r"""
    #     For the discretized survival times and censorship, we define labels and store their counts.
    #
    #     Args:
    #         - self
    #
    #     Returns:
    #         - self
    #
    #     """
    #
    #     label_dict = {}
    #     key_count = 0
    #     for i in range(len(self.bins) - 1):
    #         for c in [0, 1]:
    #             label_dict.update({(i, c): key_count})
    #             key_count += 1
    #
    #     for i in self.label_data.index:
    #         key = self.label_data.loc[i, 'label']
    #         self.label_data.at[i, 'disc_label'] = key
    #         censorship = self.label_data.loc[i, self.censorship_var]
    #         key = (key, int(censorship))
    #         self.label_data.at[i, 'label'] = label_dict[key]
    #
    #     self.num_classes = len(label_dict)
    #     self.label_dict = label_dict

    # def _get_patient_dict(self):
    #     r"""
    #     For every patient store the respective slide ids
    #
    #     Args:
    #         - self
    #
    #     Returns:
    #         - None
    #     """
    #
    #     patient_dict = {}
    #     temp_label_data = self.label_data.set_index('case_id')
    #     for patient in self.patients_df['case_id']:
    #         slide_ids = temp_label_data.loc[patient, 'slide_id']
    #         if isinstance(slide_ids, str):
    #             slide_ids = np.array(slide_ids).reshape(-1)
    #         else:
    #             slide_ids = slide_ids.values
    #         patient_dict.update({patient: slide_ids})
    #     self.patient_dict = patient_dict
    #     self.label_data = self.patients_df
    #     self.label_data.reset_index(drop=True, inplace=True)


    # def _summarize(self):
    #     r"""
    #     Summarize which type of survival you are using, number of cases and classes
    #
    #     Args:
    #         - self
    #
    #     Returns:
    #         - None
    #
    #     """
    #
    #     print("label column: {}".format(self.label_col))
    #     print("number of cases {}".format(len(self.label_data)))
    #     print("number of classes: {}".format(self.num_classes))


