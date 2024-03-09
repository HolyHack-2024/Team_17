import { Component } from '@angular/core';
import { NavigationService } from 'src/app/core/services/navigation.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {
  constructor(public navigationService: NavigationService) {
  }
  async ngOnInit() {
    this.navigationService.setShowNavbar(false);
  }

}
