import argparse, sys, os, time, datetime
from utils.utils_algo import *
from utils.utils_mlflow import make_query_from_args
from utils.models import *
from synthetic.gaussian import *
from synthetic.spiral import *
from synthetic.sinusoid import *
import mlflow
import mlflow.pytorch
import glob
from torch.nn.utils import clip_grad_norm_
import distutils.util


def get_args():
    parser = argparse.ArgumentParser(prog='ordinary_demo', usage='Demo with synthetic data.', description='description', epilog='end', add_help=True)

    parser.add_argument('-ds', '--dataset', help='dataset name', default='gaussian', type=str)
    parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=0.001, type=float)
    parser.add_argument('-bstr', '--batch_size_tr', help='mini batch size for traininig', default=200, type=int)
    parser.add_argument('-bste', '--batch_size_te', help='mini batch size for validation and test', default=500, type=int)
    parser.add_argument('-ts', '--training_samples', help='training samples per class', default=500, type=int)
    parser.add_argument('-vs', '--validation_samples', help='validation samples per class', default=50, type=int)
    parser.add_argument('-ln', '--label_noise', help='label noise', default=0.1, type=float)
    parser.add_argument('-nl', '--noise_level', help='noise level (for spiral)', default=1, type=float)
    parser.add_argument('-gn', '--gradient_norm', help='norm for gradient clipping (disabled if value is less than zero)', default=-1, type=float)
    parser.add_argument('-d', '--dimension', help='number of dimensions', default=10, type=int)
    parser.add_argument('-m', '--model', help='model name in torchvision', type=str, default='mlp_model')
    parser.add_argument('-md', '--middle_dim', help='middle dim of model', type=int, default=100)
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=500)
    parser.add_argument('-wd', '--weight_decay', help='weight decay', type=float, default=0)  
    parser.add_argument('-mm', '--momentum', help='momentum', type=float, default=0)  
    parser.add_argument('-fl', '--flood_level', help='loss threshold used for flooding', type=float, default=0.)
    parser.add_argument('-ngm', '--negative_gaussian_mean', help='negative gaussian mean', default=1, type=float)
    parser.add_argument('-rs', '--random_seed', help='set random seed', default=0, type=int)
    parser.add_argument('-tg', '--tags', help='experiment tags. Format: "key1:value1,key2:value2[,...]"', type=str)
    parser.add_argument('-lb', '--labels', help='comma-separated experiment labels that will be the keys of experiment tags with value=true.', type=str)
    parser.add_argument('-opt', '--optimizer', help='optimizer name', type=str, default='sgd')
    parser.add_argument('-sm', '--save_model', help='save models at the beginning of flooding, with best validation accuracy, and after the final epoch.', 
                        type=str, default='False')
    parser.add_argument('-gpu', '--gpu_id', help='gpu id', type=int, default='0')
    args = parser.parse_args()
    
    return args

       
def main(args):    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)            
    
    print('tr: {}, va: {}'.format(args.training_samples, args.validation_samples))    
    print('ds: {}, ln: {}'.format(args.dataset, args.label_noise))    
    
    ngm_string = "{:f}".format(args.negative_gaussian_mean)
    K = 2 # number of classes    
            
    train_loader, vali_loader, test_loader, train_cl_loader, vali_cl_loader, test_cl_loader = get_dataloader(args)      
    model = get_model(args, K)
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = get_optimizer(args, model)

    save_table = np.zeros(shape=(args.epochs, 11))
    ce_mean = nn.CrossEntropyLoss(reduction='mean')    

    for param in optimizer.param_groups:
        current_lr = param["lr"]
        
    mlflow.set_tags(dict([kw.split(':') for kw in args.tags.split(',')])) if len(args.tags) > 0 else None
    mlflow.set_tags({l: True for l in args.labels.split(',')})            

    for epoch in range(-1, args.epochs):
        flooded_count, mini_batch_count = 0, 0
        model.train()
        if epoch != -1: # skip epoch 0 for training
            for (images, labels, _) in train_loader:
                images, labels = images.to(device), labels.to(device)            
                outputs = model(images)
                loss_mean = ce_mean(outputs, labels)
                if args.flood_level > 0:
                    loss_corrected = (loss_mean-args.flood_level).abs() + args.flood_level
                else:
                    loss_corrected = loss_mean
                
                if loss_corrected != loss_mean:
                    flooded_count += 1
                mini_batch_count += 1
                optimizer.zero_grad() # Clear gradients w.r.t. parameters
                loss_corrected.backward() # backprop.
                if args.gradient_norm > 0:
                    clip_grad_norm_(model.parameters(), args.gradient_norm)
                optimizer.step()
                
        for param in optimizer.param_groups:
            current_lr = param["lr"]            
        
            
        tr_acc, tr_loss, va_acc, va_loss, te_acc, te_loss = get_acc_loss(train_loader, vali_loader, test_loader, model, device)            
        tr_cl_acc, tr_cl_loss, va_cl_acc, va_cl_loss, te_cl_acc, te_cl_loss \
            = get_acc_loss(train_cl_loader, vali_cl_loader, test_cl_loader, model, device)                    
                
        proportion = float(flooded_count)/mini_batch_count if mini_batch_count != 0 else 0

        print('flood: {} rs: {}'.format(args.flood_level, args.random_seed))
        print('Epoch: {} LR: {} TrLss: {:.4g} VaLss: {:.4g} TeLss: {:.4g} TrAcc: {:.3g} VaAcc: {:.3g} TeAcc: {:.3g}'.format(
            epoch+1, current_lr, tr_loss, va_loss, te_loss, tr_acc, va_acc, te_acc))
        print('TrClLss: {:.4g} VaClLss: {:.4g} TeClLss: {:.4g} TrClAcc: {:.3g} VaClAcc: {:.3g} TeAcc: {:.3g}'.format(
            tr_cl_loss, va_cl_loss, te_cl_loss, tr_cl_acc, va_cl_acc, te_cl_acc))
        print('Flood prop: {:.4g}'.format(proportion))        
            
        mlflow.log_metrics(step=epoch+1, metrics={
            'currentLr': current_lr,
            'trLss': tr_loss, 'vaLss': va_loss, 'teLss': te_loss,
            'trAcc': tr_acc, 'vaAcc': va_acc, 'teAcc': te_acc,
            'trclLss': tr_cl_loss, 'vaclLss': va_cl_loss, 'teclLss': te_cl_loss,
            'trclAcc': tr_cl_acc, 'vaclAcc': va_cl_acc, 'teclAcc': te_cl_acc,    
            'floodProp': proportion})               

        
def get_optimizer(args, model):
    if args.optimizer == 'sgd':        
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, 
                                    lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)    
    else:
        raise RuntimeError("Optimizer name is invalid.")    
    return optimizer
    

def get_dataloader(args):
    if args.dataset == 'gaussian':
        dataset = GaussianDataNumpy(mu_pos=np.ones(args.dimension)*0, mu_neg=np.ones(args.dimension)*args.negative_gaussian_mean, 
                              cov_pos=np.identity(10), cov_neg=np.identity(10), n_pos_tr=args.training_samples, n_neg_tr=args.training_samples,
                              n_pos_va=args.validation_samples, n_neg_va=args.validation_samples, label_noise=args.label_noise)
    elif args.dataset == 'spiral':
        dataset = SpiralDataNumpy(label_noise=args.label_noise, noise_level=args.noise_level,
                            n_pos_tr=args.training_samples, n_neg_tr=args.training_samples, 
                            n_pos_va=args.validation_samples, n_neg_va=args.validation_samples)        
    elif args.dataset == 'sinusoid':
        dataset = SinusoidDataNumpy(dim=args.dimension, label_noise=args.label_noise, n_tr=args.training_samples, n_va=args.validation_samples)  
    elif args.dataset == 'sinusoid2d':
        dataset = Sinusoid2dimDataNumpy(label_noise=args.label_noise, n_tr=args.training_samples, n_va=args.validation_samples)        
    else:
        raise RuntimeError("Dataset name is invalid.")    
        
    x_tr, r_tr, y_tr, y_tr_cl, x_va, r_va, y_va, y_va_cl, x_te, r_te, y_te, y_te_cl = dataset.makeData()    
    tr_dataset, va_dataset, te_dataset = MyDataset(x_tr, y_tr, r_tr), MyDataset(x_va, y_va, r_va), MyDataset(x_te, y_te, r_te)
    tr_cl_dataset, va_cl_dataset, te_cl_dataset = MyDataset(x_tr, y_tr_cl, r_tr), MyDataset(x_va, y_va_cl, r_va), MyDataset(x_te, y_te_cl, r_te)    
    
    train_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch_size_tr, shuffle=True, pin_memory=True)
    vali_loader = torch.utils.data.DataLoader(va_dataset, batch_size=args.batch_size_te, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(te_dataset, batch_size=args.batch_size_te, pin_memory=True)

    train_cl_loader = torch.utils.data.DataLoader(tr_cl_dataset, batch_size=args.batch_size_tr, shuffle=True, pin_memory=True)
    vali_cl_loader = torch.utils.data.DataLoader(va_cl_dataset, batch_size=args.batch_size_te, pin_memory=True)
    test_cl_loader = torch.utils.data.DataLoader(te_cl_dataset, batch_size=args.batch_size_te, pin_memory=True)    
    
    return train_loader, vali_loader, test_loader, train_cl_loader, vali_cl_loader, test_cl_loader


def get_model(args, K):
    if '_model' in args.model:
        model = globals()[args.model](input_dim=args.dimension, output_dim=K)
    else:
        model = models.__dict__[args.model](pretrained=False, num_classes=K)
        
    return model


def get_acc_loss(train_loader, vali_loader, test_loader, model, device):    
    train_acc, train_loss = acc_check(loader=train_loader, model=model, device=device)
    vali_acc, vali_loss = acc_check(loader=vali_loader, model=model, device=device)
    test_acc, test_loss = acc_check(loader=test_loader, model=model, device=device)    
    return train_acc, train_loss.item(), vali_acc, vali_loss.item(), test_acc, test_loss.item()


def get_gradients(train_loader, vali_loader, test_loader, model, device):
    tr_grad_norm = get_grad_norm(loader=train_loader, model=model, device=device)
    va_grad_norm = get_grad_norm(loader=vali_loader, model=model, device=device)
    te_grad_norm = get_grad_norm(loader=test_loader, model=model, device=device)
    tr_fngrad_norm = get_fngrad_norm(loader=train_loader, model=model, device=device)
    va_fngrad_norm = get_fngrad_norm(loader=vali_loader, model=model, device=device)
    te_fngrad_norm = get_fngrad_norm(loader=test_loader, model=model, device=device)    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)        
    return tr_grad_norm.item(), va_grad_norm.item(), te_grad_norm.item(), \
        tr_fngrad_norm.item(), va_fngrad_norm.item(), te_fngrad_norm.item(), \
        n_params


if __name__ == "__main__":                   
    args = get_args()                
    with mlflow.start_run() as run:        
        main(args)