import torch

from encoding_custom.nn.adversarial_attacks import attack_utils


class PGDAttack(attack_utils.BaseAttacker):

    def __init__(self, args, cfg, model, tasks, dl_val, device, **kwargs):
        super(PGDAttack, self).__init__(
            args, cfg, model, tasks, dl_val, device, **kwargs)
        self.using_noise = kwargs.get('using_noise', True)
        self.attack_loss = kwargs.get('attack_loss', 'segment')
        if self.attack_loss == 'segment':
            self.mtl_loss.task_to_fn = {
                'segment': self.mtl_loss.task_to_fn['segment']}
        elif self.attack_loss == 'depth':
            self.mtl_loss.task_to_fn = {
                'depth': self.mtl_loss.task_to_fn['depth']}
        elif self.attack_loss == 'sem_cont':
            self.mtl_loss.task_to_fn = {
                'sem_cont': self.mtl_loss.task_to_fn['sem_cont']}
        elif self.attack_loss == 'sur_nor':
            self.mtl_loss.task_to_fn = {
                'sur_nor': self.mtl_loss.task_to_fn['sur_nor']}
        elif self.attack_loss == 'ae':
            self.mtl_loss.task_to_fn = {
                'ae': self.mtl_loss.task_to_fn['ae']}

    def preprocess_targets(self, images, targets):
        return targets

    def preprocess_images(self, adv_images, *arguments):
        lower_bound, upper_bound = arguments
        if self.using_noise:
            epsilon = self.eps / 255.
            noise = torch.FloatTensor(adv_images.size()).uniform_(
                -epsilon, epsilon)
            noise = noise.cuda()
            noise = noise / self.std_ts
            adv_images = adv_images + noise
            return attack_utils.clamp_tensor(
                adv_images, lower_bound, upper_bound)

    def attack_objective(self, predictions, targets):
        loss, loss_dict = self.mtl_loss(predictions, targets)
        return loss
